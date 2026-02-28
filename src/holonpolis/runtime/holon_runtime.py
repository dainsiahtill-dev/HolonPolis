"""Holon Runtime - generic execution engine for any Holon.

A HolonRuntime is instantiated from a Blueprint and runs the conversation loop.
There are no specific Agent classes - all Holons are runtime instances.

Self-Evolution Capabilities:
- Request evolution of new skills through RGV (Red-Green-Verify)
- Learn from failures and automatically improve
- Compose new skills from existing ones
- Self-analyze performance and identify improvement areas
"""

import asyncio
import importlib.util
import inspect
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from holonpolis.services.evolution_service import EvolutionResult

import structlog

from holonpolis.config import settings
from holonpolis.domain import Blueprint
from holonpolis.domain.memory import MemoryKind
from holonpolis.domain.skills import SkillManifest
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.kernel.llm.llm_runtime import LLMConfig, LLMMessage, get_llm_runtime
from holonpolis.kernel.storage import HolonPathGuard
from holonpolis.kernel.tools import (
    ToolExecutor,
    ToolChainStep,
    ToolChainResult,
    execute_tool,
)
from holonpolis.services.holon_service import HolonService
from holonpolis.services.memory_service import MemoryService

logger = structlog.get_logger()


class EvolutionStatus(Enum):
    """Status of an evolution request."""
    PENDING = "pending"
    EVOLVING = "evolving"
    RED_PHASE = "red_phase"
    GREEN_PHASE = "green_phase"
    VERIFY_PHASE = "verify_phase"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class EvolutionRequest:
    """A request for skill evolution."""
    request_id: str
    holon_id: str
    skill_name: str
    description: str
    requirements: List[str]
    test_cases: List[Dict[str, Any]]
    parent_skills: List[str]  # Skills to build upon
    status: EvolutionStatus
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class HolonState:
    """Current state of a Holon conversation."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    episode_count: int = 0
    total_tokens: int = 0
    skills: List[str] = field(default_factory=list)  # IDs of evolved skills
    evolution_requests: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedSkill:
    """Callable handle for an evolved skill."""

    skill_id: str
    name: str
    description: str
    version: str
    tool_schema: Dict[str, Any]
    _executor: Callable[[Dict[str, Any]], Awaitable[Any]] = field(repr=False)

    async def execute(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        params: Dict[str, Any] = {}
        if payload is not None:
            if not isinstance(payload, dict):
                raise TypeError("payload must be a dict when provided")
            params.update(payload)
        params.update(kwargs)
        return await self._executor(params)


class CapabilityDeniedError(PermissionError):
    """Raised when a runtime capability is blocked by boundary policy."""


class SkillPayloadValidationError(ValueError):
    """Raised when input payload does not satisfy skill tool schema."""


class HolonRuntime:
    """Generic runtime for any Holon.

    Instantiated from a Blueprint, this handles:
    - Conversation loop
    - Memory recall/store
    - Tool invocation
    - Episode recording
    """

    def __init__(self, holon_id: str, blueprint: Optional[Blueprint] = None, workspace: Optional[str] = None):
        self.holon_id = holon_id
        self.blueprint = blueprint or self._load_blueprint()
        self.memory = MemoryService(holon_id)
        self.llm = get_llm_runtime()
        self.state = HolonState()
        self._evolution_requests: Dict[str, EvolutionRequest] = {}

        # Initialize tool executor with blueprint permissions
        ws = workspace or str(getattr(settings, 'workspace_root', '.'))
        self.tool_executor = ToolExecutor(
            workspace=ws,
            allow_write=self.blueprint.boundary.allow_file_write,
            allow_exec=self.blueprint.boundary.allow_subprocess,
        )

    def _load_blueprint(self) -> Blueprint:
        """Load blueprint from disk."""
        svc = HolonService()
        return svc.get_blueprint(self.holon_id)

    def _build_system_prompt(self) -> str:
        """Build the system prompt from blueprint."""
        available_tools = self.get_available_tools()
        lines = [
            f"You are {self.blueprint.name}, a specialized AI assistant.",
            "",
            f"Your purpose: {self.blueprint.purpose}",
            "",
            "# Your Capabilities",
            f"Available tools: {', '.join(available_tools) or 'None configured'}",
            "",
            "# Constraints",
            f"- File write access: {'Yes' if self.blueprint.boundary.allow_file_write else 'No'}",
            f"- Network access: {'Yes' if self.blueprint.boundary.allow_network else 'No'}",
            f"- Subprocess access: {'Yes' if self.blueprint.boundary.allow_subprocess else 'No'}",
            "",
            "You can recall relevant memories to help with the user's request.",
            "If you need a new capability, you can request evolution.",
        ]
        return "\n".join(lines)

    async def chat(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a user message and return a response.

        This is the main entry point for Holon interaction.

        Args:
            user_message: The user's message
            conversation_id: Optional conversation ID for continuity

        Returns:
            Response dict with content and metadata
        """
        start_time = time.time()

        # 1. Recall relevant memories
        memories = await self.memory.recall(
            query=user_message,
            top_k=5,
        )

        # 2. Build LLM messages
        system_prompt = self._build_system_prompt()

        # Add memory context if available
        memory_context = ""
        if memories:
            memory_context = "\n\n# Relevant Memories\n" + "\n".join(
                f"- [{m['kind']}] {m['content'][:200]}"
                for m in memories
            )

        messages = [
            LLMMessage(role="system", content=system_prompt + memory_context),
        ]

        # Add conversation history
        for msg in self.state.messages[-10:]:  # Last 10 messages
            messages.append(LLMMessage(
                role=msg["role"],
                content=msg["content"],
            ))

        # Add current user message
        messages.append(LLMMessage(role="user", content=user_message))

        # 3. Call LLM
        # Map provider to correct model
        provider_model_map = {
            "minimax": getattr(settings, "minimax_model", "MiniMax-M2.5"),
            "kimi": getattr(settings, "kimi_model", "moonshot-v1-8k"),
            "kimi-coding": "kimi-for-coding",
            "ollama": getattr(settings, "ollama_model", "qwen2.5-coder:14b"),
            "ollama-local": getattr(settings, "ollama_model", "qwen2.5-coder:14b"),
            "openai": getattr(settings, "openai_model", "gpt-4o-mini"),
            "anthropic": getattr(settings, "anthropic_model", "claude-3-5-sonnet-latest"),
        }
        default_model = provider_model_map.get(settings.llm_provider, settings.openai_model)

        config = LLMConfig(
            provider_id=settings.llm_provider,
            model=default_model,
            temperature=settings.llm_temperature,
            max_tokens=min(
                self.blueprint.boundary.max_tokens_per_episode,
                settings.llm_max_tokens,
            ),
        )

        try:
            response = await self.llm.chat_with_history(messages, config=config)

            latency_ms = int((time.time() - start_time) * 1000)

            # 4. Update state
            self.state.messages.append({"role": "user", "content": user_message})
            self.state.messages.append({"role": "assistant", "content": response.content})
            self.state.episode_count += 1

            # 5. Record episode
            tool_chain = self.state.context.get("last_tool_chain", [])
            episode_id = await self.memory.write_episode(
                transcript=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response.content},
                ],
                tool_chain=tool_chain,
                outcome="success",
                cost=response.usage.total_tokens * 0.00001,  # Rough estimate
                latency_ms=latency_ms,
            )

            # 6. Condense to memory if significant
            if len(user_message) > 50 or len(response.content) > 100:
                await self._condense_to_memory(
                    episode_id=episode_id,
                    user_message=user_message,
                    response=response.content,
                )

            logger.debug(
                "holon_chat_complete",
                holon_id=self.holon_id,
                episode_id=episode_id,
                latency_ms=latency_ms,
            )

            return {
                "content": response.content,
                "holon_id": self.holon_id,
                "episode_id": episode_id,
                "latency_ms": latency_ms,
                "memories_recalled": len(memories),
            }

        except Exception as e:
            logger.error("holon_chat_failed", holon_id=self.holon_id, error=str(e))

            # Record failure episode
            await self.memory.write_episode(
                transcript=[
                    {"role": "user", "content": user_message},
                    {"role": "system", "content": f"Error: {str(e)}"},
                ],
                outcome="failure",
                outcome_details={"error": str(e)},
                latency_ms=int((time.time() - start_time) * 1000),
            )

            raise

    async def _condense_to_memory(
        self,
        episode_id: str,
        user_message: str,
        response: str,
    ) -> None:
        """Condense an interaction into a retrievable memory."""
        # Simple extraction - in production, this might use LLM summarization
        summary = f"User asked: {user_message[:100]}... Response: {response[:100]}..."

        await self.memory.consolidate_episode_to_memory(
            episode_id=episode_id,
            summary=summary,
            tags=["conversation", "auto_extracted"],
            importance=0.5,
        )

    async def remember(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
    ) -> str:
        """Explicitly remember something.

        This can be called by the Holon or external systems.
        """
        return await self.memory.remember(
            content=content,
            kind=MemoryKind.FACT,
            tags=tags or [],
            importance=importance,
        )

    async def recall(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recall memories relevant to a query."""
        return await self.memory.recall(query, top_k=top_k)

    async def request_evolution(
        self,
        skill_name: str,
        description: str,
        requirements: List[str],
        test_cases: Optional[List[Dict[str, Any]]] = None,
        parent_skills: Optional[List[str]] = None,
    ) -> EvolutionRequest:
        """Request evolution of a new skill through RGV (Red-Green-Verify).

        This initiates the full evolution pipeline:
        1. Red: Generate test cases that define expected behavior
        2. Green: Generate code to pass the tests
        3. Verify: Security scan and validation
        4. Persist: Save skill to local skills directory

        Args:
            skill_name: Name for the new skill
            description: What the skill should do
            requirements: List of functional requirements
            test_cases: Optional predefined test cases
            parent_skills: Optional skills to build upon

        Returns:
            EvolutionRequest with status and result
        """
        request_id = f"evo_{uuid.uuid4().hex[:12]}"

        logger.info(
            "evolution_requested",
            holon_id=self.holon_id,
            skill_name=skill_name,
            request_id=request_id,
        )

        # Create evolution request
        request = EvolutionRequest(
            request_id=request_id,
            holon_id=self.holon_id,
            skill_name=skill_name,
            description=description,
            requirements=requirements,
            test_cases=test_cases or [],
            parent_skills=parent_skills or [],
            status=EvolutionStatus.PENDING,
            created_at=utc_now_iso(),
        )

        # Store request
        self.state.evolution_requests.append(request_id)
        self._evolution_requests[request_id] = request
        await self.remember(
            content=f"Evolution request {request_id}: {skill_name} - {description}",
            tags=["evolution", "skill_request"],
            importance=0.9,
        )

        # Execute evolution asynchronously
        asyncio.create_task(self._execute_evolution(request))

        return request

    async def _execute_evolution(self, request: EvolutionRequest) -> None:
        """Execute the RGV evolution pipeline."""
        from holonpolis.domain.skills import ToolSchema
        from holonpolis.services.evolution_service import EvolutionService
        from holonpolis.genesis.genesis_memory import GenesisMemory

        result: Optional["EvolutionResult"] = None
        try:
            request.status = EvolutionStatus.EVOLVING
            logger.info("evolution_started", request_id=request.request_id)
            genesis_memory = GenesisMemory()

            # Get evolution service
            evolution_service = EvolutionService()

            # Red Phase: Generate or validate test cases
            request.status = EvolutionStatus.RED_PHASE

            # Build tool schema for the skill
            tool_schema = ToolSchema(
                name=request.skill_name,
                description=request.description,
                parameters={
                    "type": "object",
                    "properties": {},
                },
            )
            if request.test_cases:
                request.status = EvolutionStatus.GREEN_PHASE
                result = await evolution_service.evolve_skill_with_test_cases(
                    holon_id=self.holon_id,
                    skill_name=request.skill_name,
                    description=request.description,
                    requirements=request.requirements,
                    test_cases=request.test_cases,
                    tool_schema=tool_schema,
                )
            else:
                # Autonomous mode: LLM generates tests + code inside EvolutionService.
                result = await evolution_service.evolve_skill_autonomous(
                    holon_id=self.holon_id,
                    skill_name=request.skill_name,
                    description=request.description,
                    requirements=request.requirements,
                    tool_schema=tool_schema,
                )

        except Exception as e:
            request.status = EvolutionStatus.FAILED
            request.error_message = str(e)
            logger.error(
                "evolution_failed",
                request_id=request.request_id,
                error=str(e),
            )
            request.completed_at = utc_now_iso()
            return

        if result and result.success:
            request.status = EvolutionStatus.COMPLETED
            request.result = {
                "skill_id": result.skill_id,
                "attestation_id": result.attestation.attestation_id if result.attestation else None,
                "code_path": result.code_path,
            }
            if result.skill_id:
                self.state.skills.append(result.skill_id)

            await self._safe_remember(
                content=f"Successfully evolved skill: {request.skill_name} ({result.skill_id})",
                tags=["evolution", "skill_completed"],
                importance=1.0,
            )
            await self._safe_record_evolution(
                genesis_memory=genesis_memory,
                skill_name=request.skill_name,
                status="success",
                attestation_id=result.attestation.attestation_id if result.attestation else None,
            )

            logger.info(
                "evolution_completed",
                request_id=request.request_id,
                skill_id=result.skill_id,
            )
            request.completed_at = utc_now_iso()
            return

        request.status = EvolutionStatus.FAILED
        request.error_message = result.error_message if result else "evolution_result_missing"
        await self._safe_record_evolution(
            genesis_memory=genesis_memory,
            skill_name=request.skill_name,
            status="failed",
            error_message=request.error_message,
        )
        await self._safe_learn_from_evolution_failure(request, result)
        request.completed_at = utc_now_iso()

    async def _safe_record_evolution(
        self,
        genesis_memory: Any,
        skill_name: str,
        status: str,
        attestation_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record evolution audit data without mutating final RGV status on failure."""
        try:
            await genesis_memory.record_evolution(
                holon_id=self.holon_id,
                skill_name=skill_name,
                status=status,
                attestation_id=attestation_id,
                error_message=error_message,
            )
        except Exception as exc:
            logger.warning(
                "genesis_evolution_audit_failed",
                holon_id=self.holon_id,
                skill_name=skill_name,
                status=status,
                error=str(exc),
            )

    async def _safe_remember(
        self,
        content: str,
        tags: List[str],
        importance: float,
    ) -> None:
        """Best-effort memory write for non-critical telemetry."""
        try:
            await self.remember(content=content, tags=tags, importance=importance)
        except Exception as exc:
            logger.warning(
                "evolution_memory_write_failed",
                holon_id=self.holon_id,
                error=str(exc),
            )

    async def _safe_learn_from_evolution_failure(
        self,
        request: EvolutionRequest,
        result: Optional["EvolutionResult"],
    ) -> None:
        """Best-effort post-mortem learning for failed evolutions."""
        if result is None:
            return
        try:
            await self._learn_from_evolution_failure(request, result)
        except Exception as exc:
            logger.warning(
                "evolution_failure_learning_failed",
                holon_id=self.holon_id,
                request_id=request.request_id,
                error=str(exc),
            )

    async def _learn_from_evolution_failure(
        self,
        request: EvolutionRequest,
        result: Any,
    ) -> None:
        """Learn from evolution failure and potentially retry."""
        logger.info(
            "learning_from_evolution_failure",
            request_id=request.request_id,
            phase=result.phase if hasattr(result, 'phase') else 'unknown',
        )

        await self.remember(
            content=f"Evolution failed for {request.skill_name}: {result.error_message}",
            tags=["evolution", "failure", request.skill_name],
            importance=0.8,
        )

    async def self_improve(self) -> Dict[str, Any]:
        """Analyze own performance and suggest improvements.

        This method allows a Holon to:
        1. Analyze its success/failure patterns
        2. Identify gaps in its capabilities
        3. Request evolution of new skills to fill gaps

        Returns:
            Improvement plan with suggested evolutions
        """
        # Analyze recent episodes
        recent_episodes = await self.memory.get_episodes(limit=100)

        # Calculate metrics
        total = len(recent_episodes)
        if total == 0:
            return {"status": "no_data", "suggestions": []}

        successes = sum(1 for e in recent_episodes if e.get('outcome') == 'success')
        failures = sum(1 for e in recent_episodes if e.get('outcome') == 'failure')
        success_rate = successes / total if total > 0 else 0

        # Identify common failure patterns
        failure_patterns = {}
        for ep in recent_episodes:
            if ep.get('outcome') == 'failure':
                error = ep.get('outcome_details', {}).get('error', 'unknown')
                failure_patterns[error] = failure_patterns.get(error, 0) + 1

        # Generate improvement suggestions
        suggestions = []

        if success_rate < 0.8:
            suggestions.append({
                "type": "evolve_skill",
                "reason": f"Low success rate ({success_rate:.1%})",
                "suggested_skill": "error_recovery",
            })

        if failures > 10:
            suggestions.append({
                "type": "improve_memory",
                "reason": "High failure count suggests memory gaps",
                "action": "consolidate_more_memories",
            })

        # Store self-analysis
        await self.remember(
            content=f"Self-improvement analysis: {success_rate:.1%} success rate, {len(suggestions)} suggestions",
            tags=["self_improvement", "analysis"],
            importance=0.9,
        )

        return {
            "status": "analyzed",
            "metrics": {
                "total_episodes": total,
                "success_rate": success_rate,
                "failure_patterns": failure_patterns,
            },
            "suggestions": suggestions,
        }

    async def compose_skill(
        self,
        new_skill_name: str,
        parent_skill_ids: List[str],
        composition_description: str,
    ) -> EvolutionRequest:
        """Compose a new skill from existing skills.

        This allows Holons to build higher-level capabilities
        by combining existing skills.

        Args:
            new_skill_name: Name for the composed skill
            parent_skill_ids: IDs of skills to compose
            composition_description: How to combine the skills

        Returns:
            EvolutionRequest for the composed skill
        """
        # Recall parent skills
        parent_descriptions = []
        for skill_id in parent_skill_ids:
            memories = await self.recall(f"skill {skill_id}", top_k=2)
            if memories:
                parent_descriptions.append(memories[0]['content'])

        # Generate composed requirements
        requirements = [
            f"Combine skills: {', '.join(parent_skill_ids)}",
            composition_description,
            "Maintain atomicity of parent skills",
            "Provide unified interface",
        ]

        return await self.request_evolution(
            skill_name=new_skill_name,
            description=f"Composed skill: {composition_description}",
            requirements=requirements,
            parent_skills=parent_skill_ids,
        )

    def get_evolution_status(self, request_id: str) -> Optional[EvolutionRequest]:
        """Get status of an evolution request."""
        return self._evolution_requests.get(request_id)

    async def wait_for_evolution(
        self,
        request_id: str,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 0.5,
    ) -> EvolutionRequest:
        """Wait until an evolution request reaches a terminal state."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")

        deadline = time.monotonic() + timeout_seconds
        terminal = {
            EvolutionStatus.COMPLETED,
            EvolutionStatus.FAILED,
            EvolutionStatus.REJECTED,
        }

        while True:
            status = self.get_evolution_status(request_id)
            if status is None:
                raise ValueError(f"Evolution request not found: {request_id}")
            if status.status in terminal:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for evolution request {request_id} "
                    f"after {timeout_seconds:.1f}s"
                )
            await asyncio.sleep(poll_interval_seconds)

    def list_skills(self) -> List[Dict[str, Any]]:
        """List persisted evolved skills for this Holon."""
        manifests: Dict[str, Dict[str, Any]] = {}
        for skill_dir in self._iter_skill_dirs():
            manifest_path = skill_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = SkillManifest.from_dict(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue

            manifests[manifest.skill_id] = {
                "skill_id": manifest.skill_id,
                "name": manifest.name,
                "description": manifest.description,
                "version": manifest.version,
                "path": str(skill_dir),
            }

        return sorted(manifests.values(), key=lambda item: item["name"].lower())

    def get_skill(self, skill_name_or_id: str) -> LoadedSkill:
        """Load an evolved skill by skill_id or skill name."""
        skill_dir = self._resolve_skill_directory(skill_name_or_id)
        if skill_dir is None:
            raise FileNotFoundError(f"Skill '{skill_name_or_id}' not found")

        manifest = self._read_skill_manifest(skill_dir)
        latest = manifest.versions[-1] if manifest.versions else None
        if latest is not None and not latest.static_scan_passed:
            raise RuntimeError(
                f"Skill '{manifest.name}' latest version failed static verification"
            )

        module = self._load_skill_module(manifest, skill_dir)
        entrypoint = self._select_skill_callable(module, manifest)

        async def _executor(params: Dict[str, Any]) -> Any:
            return await self._invoke_skill_callable(entrypoint, params)

        return LoadedSkill(
            skill_id=manifest.skill_id,
            name=manifest.name,
            description=manifest.description,
            version=manifest.version,
            tool_schema=manifest.tool_schema.to_dict(),
            _executor=_executor,
        )

    async def execute_skill(
        self,
        skill_name_or_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Convenience wrapper to execute a persisted evolved skill."""
        self.enforce_capability("skill.execute", aliases=["execute"])
        skill = self.get_skill(skill_name_or_id)
        merged_payload: Dict[str, Any] = {}
        if payload is not None:
            if not isinstance(payload, dict):
                raise SkillPayloadValidationError("payload must be an object/dict")
            merged_payload.update(payload)
        merged_payload.update(kwargs)
        self._validate_payload_against_tool_schema(merged_payload, skill.tool_schema)
        return await skill.execute(payload=merged_payload)

    def _validate_payload_against_tool_schema(
        self,
        payload: Dict[str, Any],
        tool_schema: Dict[str, Any],
    ) -> None:
        """Strictly validate payload against manifest tool schema."""
        if not isinstance(payload, dict):
            raise SkillPayloadValidationError("payload must be an object/dict")
        if not isinstance(tool_schema, dict):
            return

        parameters = tool_schema.get("parameters", {})
        if not isinstance(parameters, dict):
            return

        root_schema: Dict[str, Any] = dict(parameters)
        if "type" not in root_schema and (
            "properties" in root_schema or "required" in root_schema
        ):
            root_schema["type"] = "object"

        manifest_required = tool_schema.get("required", [])
        if isinstance(manifest_required, list) and manifest_required and "required" not in root_schema:
            root_schema["required"] = manifest_required

        self._validate_json_schema_value(
            value=payload,
            schema=root_schema,
            path="payload",
            strict_object_default=True,
        )

    def _validate_json_schema_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        strict_object_default: bool,
    ) -> None:
        expected_type = schema.get("type")
        if expected_type is not None:
            allowed_types = (
                expected_type if isinstance(expected_type, list) else [expected_type]
            )
            if not any(self._value_matches_type(value, t) for t in allowed_types):
                expected = " | ".join(str(t) for t in allowed_types)
                actual = self._infer_json_type(value)
                raise SkillPayloadValidationError(
                    f"{path} expected type {expected}, got {actual}"
                )

        if "enum" in schema and value not in schema["enum"]:
            raise SkillPayloadValidationError(
                f"{path} must be one of {schema['enum']}, got {value!r}"
            )
        if "const" in schema and value != schema["const"]:
            raise SkillPayloadValidationError(
                f"{path} must equal {schema['const']!r}, got {value!r}"
            )

        if isinstance(value, str):
            min_length = schema.get("minLength")
            max_length = schema.get("maxLength")
            pattern = schema.get("pattern")
            if isinstance(min_length, int) and len(value) < min_length:
                raise SkillPayloadValidationError(f"{path} length must be >= {min_length}")
            if isinstance(max_length, int) and len(value) > max_length:
                raise SkillPayloadValidationError(f"{path} length must be <= {max_length}")
            if isinstance(pattern, str) and re.search(pattern, value) is None:
                raise SkillPayloadValidationError(f"{path} does not match pattern {pattern!r}")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            exclusive_min = schema.get("exclusiveMinimum")
            exclusive_max = schema.get("exclusiveMaximum")
            if minimum is not None and value < minimum:
                raise SkillPayloadValidationError(f"{path} must be >= {minimum}")
            if maximum is not None and value > maximum:
                raise SkillPayloadValidationError(f"{path} must be <= {maximum}")
            if exclusive_min is not None and value <= exclusive_min:
                raise SkillPayloadValidationError(f"{path} must be > {exclusive_min}")
            if exclusive_max is not None and value >= exclusive_max:
                raise SkillPayloadValidationError(f"{path} must be < {exclusive_max}")

        if isinstance(value, list):
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if isinstance(min_items, int) and len(value) < min_items:
                raise SkillPayloadValidationError(f"{path} must contain at least {min_items} items")
            if isinstance(max_items, int) and len(value) > max_items:
                raise SkillPayloadValidationError(f"{path} must contain at most {max_items} items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    self._validate_json_schema_value(
                        value=item,
                        schema=item_schema,
                        path=f"{path}[{index}]",
                        strict_object_default=strict_object_default,
                    )

        if isinstance(value, dict):
            required = schema.get("required", [])
            if isinstance(required, list):
                missing = [key for key in required if key not in value]
                if missing:
                    raise SkillPayloadValidationError(
                        f"{path} missing required fields: {', '.join(missing)}"
                    )

            properties = schema.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            additional = schema.get("additionalProperties", None)
            additional_schema = additional if isinstance(additional, dict) else None
            if isinstance(additional, bool):
                additional_allowed = additional
            elif additional is None:
                additional_allowed = False if (strict_object_default and properties) else True
            else:
                additional_allowed = True

            for key, item in value.items():
                child_path = f"{path}.{key}"
                if key in properties and isinstance(properties[key], dict):
                    self._validate_json_schema_value(
                        value=item,
                        schema=properties[key],
                        path=child_path,
                        strict_object_default=strict_object_default,
                    )
                elif additional_schema is not None:
                    self._validate_json_schema_value(
                        value=item,
                        schema=additional_schema,
                        path=child_path,
                        strict_object_default=strict_object_default,
                    )
                elif not additional_allowed:
                    raise SkillPayloadValidationError(f"{path} contains unexpected field: {key}")

    @staticmethod
    def _value_matches_type(value: Any, schema_type: Any) -> bool:
        if schema_type == "string":
            return isinstance(value, str)
        if schema_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if schema_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if schema_type == "boolean":
            return isinstance(value, bool)
        if schema_type == "object":
            return isinstance(value, dict)
        if schema_type == "array":
            return isinstance(value, list)
        if schema_type == "null":
            return value is None
        return True

    @staticmethod
    def _infer_json_type(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, dict):
            return "object"
        if isinstance(value, list):
            return "array"
        return type(value).__name__

    def _iter_skill_roots(self) -> List[Path]:
        guard = HolonPathGuard(self.holon_id)
        roots: List[Path] = []
        for rel_path in ("skills_local", "skills"):
            root = guard.resolve(rel_path, must_exist=False)
            if root.exists() and root.is_dir():
                roots.append(root)
        return roots

    def _iter_skill_dirs(self) -> List[Path]:
        skill_dirs: List[Path] = []
        for root in self._iter_skill_roots():
            for child in root.iterdir():
                if child.is_dir():
                    skill_dirs.append(child)
        return skill_dirs

    def _resolve_skill_directory(self, skill_name_or_id: str) -> Optional[Path]:
        slug = self._slugify(skill_name_or_id)
        for root in self._iter_skill_roots():
            candidate = root / slug
            if candidate.exists() and candidate.is_dir():
                return candidate

        needle = skill_name_or_id.strip().lower()
        for skill_dir in self._iter_skill_dirs():
            manifest_path = skill_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = SkillManifest.from_dict(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
            if manifest.skill_id.lower() == needle or manifest.name.strip().lower() == needle:
                return skill_dir
        return None

    def _read_skill_manifest(self, skill_dir: Path) -> SkillManifest:
        manifest_path = skill_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SkillManifest.from_dict(data)

    def _resolve_skill_code_path(self, manifest: SkillManifest, skill_dir: Path) -> Path:
        guard = HolonPathGuard(self.holon_id)
        version_entry = manifest.versions[-1] if manifest.versions else None
        candidate = version_entry.code_path if version_entry else ""
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.is_absolute():
                if candidate_path.exists():
                    return candidate_path
            else:
                try:
                    resolved = guard.resolve(candidate_path)
                    if resolved.exists():
                        return resolved
                except Exception:
                    pass
        fallback = skill_dir / "skill.py"
        if not fallback.exists():
            raise FileNotFoundError(f"Skill code not found: {fallback}")
        return fallback

    def _load_skill_module(self, manifest: SkillManifest, skill_dir: Path) -> ModuleType:
        code_path = self._resolve_skill_code_path(manifest, skill_dir)
        module_key = f"holon_skill_{self.holon_id}_{manifest.skill_id}_{hash(str(code_path)) & 0xfffffff}"
        spec = importlib.util.spec_from_file_location(module_key, str(code_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load skill module from {code_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _select_skill_callable(self, module: ModuleType, manifest: SkillManifest) -> Callable[..., Any]:
        if hasattr(module, "execute") and callable(getattr(module, "execute")):
            return getattr(module, "execute")

        schema_name = manifest.tool_schema.name.strip()
        for candidate_name in (schema_name, self._slugify(schema_name), manifest.skill_id):
            if hasattr(module, candidate_name):
                candidate = getattr(module, candidate_name)
                if callable(candidate):
                    return candidate

        public_callables: List[Callable[..., Any]] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            candidate = getattr(module, name)
            if callable(candidate):
                public_callables.append(candidate)

        if len(public_callables) == 1:
            return public_callables[0]

        raise RuntimeError(
            f"Skill '{manifest.name}' does not expose a clear callable entrypoint"
        )

    async def _invoke_skill_callable(
        self,
        fn: Callable[..., Any],
        params: Dict[str, Any],
    ) -> Any:
        signature = inspect.signature(fn)
        positional_only = [
            p for p in signature.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_ONLY
        ]
        required_named = [
            p for p in signature.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and p.default is inspect.Parameter.empty
        ]
        supports_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )

        if not signature.parameters:
            result = fn()
        elif len(signature.parameters) == 1 and not supports_kwargs and not positional_only:
            only_param = next(iter(signature.parameters.values()))
            if only_param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ) and only_param.name in params and len(params) == 1:
                result = fn(**params)
            else:
                result = fn(params)
        elif positional_only:
            # Fallback for unusual signatures: pass full payload as single positional argument.
            result = fn(params)
        else:
            call_args: Dict[str, Any] = {}
            for param in signature.parameters.values():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    if param.name in params:
                        call_args[param.name] = params[param.name]
                    elif param.default is inspect.Parameter.empty:
                        raise ValueError(f"Missing required skill argument: {param.name}")
            if supports_kwargs:
                for key, value in params.items():
                    if key not in call_args:
                        call_args[key] = value
            elif any(p.name not in params for p in required_named):
                missing = [p.name for p in required_named if p.name not in params]
                raise ValueError(f"Missing required skill arguments: {', '.join(missing)}")
            result = fn(**call_args)

        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _slugify(name: str) -> str:
        import re

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return slug or "skill"

    async def execute_tool(
        self,
        tool: str,
        args: Dict[str, Any],
        *,
        act_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool through the tool executor.

        Args:
            tool: Tool name
            args: Tool arguments
            act_files: Declared scope files for write validation

        Returns:
            Tool execution result
        """
        result = self.tool_executor.execute(tool, args, act_files=act_files)

        # Record tool execution in state context
        tool_step = {
            "tool": tool,
            "args": args,
            "result": result,
        }
        if "last_tool_chain" not in self.state.context:
            self.state.context["last_tool_chain"] = []
        self.state.context["last_tool_chain"].append(tool_step)

        return result

    async def execute_tool_chain(
        self,
        steps: List[ToolChainStep],
    ) -> ToolChainResult:
        """Execute a chain of tools with dependency resolution.

        Args:
            steps: List of tool chain steps

        Returns:
            ToolChainResult with execution results
        """
        outputs: List[Dict[str, Any]] = []
        errors: List[str] = []
        saved_results: Dict[str, Any] = {}

        completed = 0
        failed = 0
        retried = 0

        for step in steps:
            # Resolve input from saved results if specified
            args = dict(step.args)
            if step.input_from and step.input_from in saved_results:
                # Merge saved result into args
                args.update(saved_results[step.input_from])

            # Execute with retry logic
            result = None
            for attempt in range(step.max_retries + 1):
                result = await self.execute_tool(step.tool, args)

                if result.get("ok"):
                    completed += 1
                    break
                elif attempt < step.max_retries and step.on_error == "retry":
                    retried += 1
                    continue
                else:
                    failed += 1
                    errors.append(f"{step.tool}: {result.get('error', 'unknown error')}")
                    if step.on_error == "stop":
                        break

            if result:
                outputs.append({
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "result": result,
                })

                # Save result if requested
                if step.save_as:
                    saved_results[step.save_as] = result

        return ToolChainResult(
            ok=failed == 0,
            outputs=outputs,
            errors=errors,
            total_steps=len(steps),
            completed_steps=completed,
            failed_steps=failed,
            retried_steps=retried,
            saved_results=saved_results,
        )

    def get_available_tools(self) -> List[str]:
        """Get list of tools available to this holon based on permissions."""
        from holonpolis.kernel.tools import supported_tool_names, read_tool_names, write_tool_names, exec_tool_names

        tools = set(read_tool_names())
        if self.blueprint.boundary.allow_file_write:
            tools.update(write_tool_names())
        if self.blueprint.boundary.allow_subprocess:
            tools.update(exec_tool_names())
        return sorted(tools)

    @staticmethod
    def _capability_matches_pattern(capability: str, pattern: str) -> bool:
        cap = capability.strip().lower()
        pat = pattern.strip().lower()
        if not pat:
            return False
        if pat == "*":
            return True
        if pat.endswith("*"):
            return cap.startswith(pat[:-1])
        return cap == pat

    def is_capability_allowed(self, capability: str, aliases: Optional[List[str]] = None) -> bool:
        """Check if a high-level runtime capability is allowed by blueprint boundary."""
        tokens = [capability]
        if aliases:
            tokens.extend(aliases)
        normalized = [t.strip().lower() for t in tokens if t and t.strip()]
        if not normalized:
            return True

        denied = [str(t).strip().lower() for t in self.blueprint.boundary.denied_tools if str(t).strip()]
        for token in normalized:
            if any(self._capability_matches_pattern(token, pattern) for pattern in denied):
                return False

        allowed = [str(t).strip().lower() for t in self.blueprint.boundary.allowed_tools if str(t).strip()]
        if not allowed:
            return True

        return any(
            self._capability_matches_pattern(token, pattern)
            for token in normalized
            for pattern in allowed
        )

    def enforce_capability(self, capability: str, aliases: Optional[List[str]] = None) -> None:
        """Raise when capability is denied by boundary policy."""
        if self.is_capability_allowed(capability, aliases=aliases):
            return

        requested = [capability] + (aliases or [])
        raise CapabilityDeniedError(
            "Capability denied by boundary policy. "
            f"Requested any of: {requested}; "
            f"allowed={self.blueprint.boundary.allowed_tools}; "
            f"denied={self.blueprint.boundary.denied_tools}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            "holon_id": self.holon_id,
            "name": self.blueprint.name,
            "species": self.blueprint.species_id,
            "episode_count": self.state.episode_count,
            "total_tokens": self.state.total_tokens,
            "available_tools": len(self.get_available_tools()),
            "evolved_skills": len(self.state.skills),
            "evolution_requests": len(self.state.evolution_requests),
        }

    async def run_selection(self, threshold: float = 0.3) -> Dict[str, Any]:
        """Execute market selection and return ecosystem survival stats."""
        self.enforce_capability(
            "social.selection.execute",
            aliases=["selection", "execute"],
        )
        from holonpolis.services.market_service import MarketService

        market = MarketService()
        result = market.run_selection(threshold=threshold)
        await self.remember(
            content=(
                f"Selection executed with threshold={threshold}: "
                f"{result.get('survivors', 0)}/{result.get('total', 0)} survived"
            ),
            tags=["selection", "market"],
            importance=0.7,
        )
        return result

    def get_market_stats(self) -> Dict[str, Any]:
        """Get current aggregated market statistics."""
        from holonpolis.services.market_service import MarketService

        return MarketService().get_market_stats()

    async def register_relationship(
        self,
        target_holon_id: str,
        relationship_type: str = "peer",
        strength: float = 0.5,
        trust_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a social relationship edge and persist it."""
        from holonpolis.domain.social import RelationshipType, SocialRelationship
        from holonpolis.services.collaboration_service import CollaborationService

        rel_type = RelationshipType(relationship_type.lower())
        relationship = SocialRelationship(
            relationship_id=f"rel_{uuid.uuid4().hex[:12]}",
            source_holon=self.holon_id,
            target_holon=target_holon_id,
            rel_type=rel_type,
            strength=max(0.0, min(1.0, strength)),
            trust_score=max(0.0, min(1.0, trust_score)),
            metadata=metadata or {},
        )

        service = CollaborationService()
        service.register_relationship(relationship)

        await self.remember(
            content=(
                f"Established {rel_type.value} relationship with {target_holon_id} "
                f"(strength={relationship.strength:.2f}, trust={relationship.trust_score:.2f})"
            ),
            tags=["social", "relationship", rel_type.value],
            importance=0.7,
        )
        return relationship.relationship_id

    async def propagate_trust(self, target_holon_id: str, max_hops: int = 2) -> float:
        """Estimate trust toward another Holon through social graph propagation."""
        from holonpolis.services.collaboration_service import CollaborationService

        service = CollaborationService()
        trust = service.social_graph.propagate_trust(
            source_holon=self.holon_id,
            target_holon=target_holon_id,
            max_hops=max_hops,
        )
        await self.remember(
            content=f"Propagated trust to {target_holon_id}: {trust:.3f} (max_hops={max_hops})",
            tags=["social", "trust"],
            importance=0.5,
        )
        return trust

    # ========== Social Capabilities ==========

    async def collaborate(
        self,
        task_name: str,
        task_description: str,
        collaborator_ids: List[str],
        subtasks: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """与其他 Holon 协作完成任务.

        Args:
            task_name: 任务名称
            task_description: 任务描述
            collaborator_ids: 协作者 Holon IDs
            subtasks: 子任务列表 [{"name": "...", "description": "..."}]

        Returns:
            协作结果
        """
        from holonpolis.services.collaboration_service import CollaborationService

        service = CollaborationService()

        # 创建协作任务
        task_structure = {
            "subtasks": subtasks,
            "dependencies": {},  # 简单并行执行
        }

        all_participants = [self.holon_id] + collaborator_ids

        task = await service.create_collaboration(
            name=task_name,
            description=task_description,
            coordinator_id=self.holon_id,
            participant_ids=all_participants,
            task_structure=task_structure,
        )

        # 启动协作
        await service.start_collaboration(task.task_id)

        # 记录到记忆
        await self.remember(
            content=f"Initiated collaboration '{task_name}' with {', '.join(collaborator_ids)}",
            tags=["collaboration", "social"],
            importance=0.8,
        )

        return {
            "collaboration_id": task.task_id,
            "status": "started",
            "participants": len(all_participants),
            "subtasks": len(subtasks),
        }

    async def offer_skill(
        self,
        skill_name: str,
        description: str,
        price_per_use: float = 0.0,
    ) -> str:
        """在技能市场发布报价.

        Args:
            skill_name: 技能名称
            description: 技能描述
            price_per_use: 每次使用的价格 (tokens)

        Returns:
            offer_id: 报价 ID
        """
        from holonpolis.services.market_service import MarketService

        market = MarketService()

        offer = market.register_offer(
            holon_id=self.holon_id,
            skill_name=skill_name,
            skill_description=description,
            price_per_use=price_per_use,
            success_rate=0.9,  # 初始成功率
        )

        # 记录到记忆
        await self.remember(
            content=f"Listed skill '{skill_name}' on market for {price_per_use} tokens/use",
            tags=["market", "skill_offer"],
            importance=0.7,
        )

        logger.info(
            "skill_offered",
            holon_id=self.holon_id,
            offer_id=offer.offer_id,
            skill=skill_name,
        )

        return offer.offer_id

    async def find_skill_providers(
        self,
        skill_query: str,
        max_price: Optional[float] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """查找技能提供者.

        Args:
            skill_query: 技能搜索词
            max_price: 最高价格限制
            top_k: 返回前 K 个结果

        Returns:
            List of provider info
        """
        from holonpolis.services.market_service import MarketService

        market = MarketService()

        results = market.find_offers(
            skill_query=skill_query,
            max_price=max_price,
            top_k=top_k,
        )

        providers = []
        for offer, score in results:
            providers.append({
                "holon_id": offer.holon_id,
                "skill_name": offer.skill_name,
                "price": offer.price_per_use,
                "success_rate": offer.success_rate,
                "rating": offer.rating,
                "match_score": score,
            })

        return providers

    async def compete(
        self,
        task_description: str,
        competitors: List[str],
        evaluation_criteria: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """参与竞争评估.

        Args:
            task_description: 竞争任务描述
            competitors: 竞争者 Holon IDs (包含自己)
            evaluation_criteria: 评估标准 {维度: 权重}

        Returns:
            竞争结果
        """
        self.enforce_capability(
            "social.competition.execute",
            aliases=["competition", "execute"],
        )
        from holonpolis.services.market_service import MarketService

        market = MarketService()

        if evaluation_criteria is None:
            evaluation_criteria = {
                "accuracy": 0.4,
                "speed": 0.3,
                "quality": 0.3,
            }

        # 模拟测试用例
        test_cases = [
            {"input": f"Task: {task_description}", "expected": "success"},
        ]

        result = await market.run_competition(
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            participant_ids=competitors,
            test_cases=test_cases,
        )

        # 记录结果
        my_rank = result.ranking.index(self.holon_id) + 1 if self.holon_id in result.ranking else None

        await self.remember(
            content=f"Competed in '{task_description[:50]}...' ranked #{my_rank}, reward: {result.rewards.get(self.holon_id, 0)}",
            tags=["competition", "social"],
            importance=0.9 if my_rank == 1 else 0.6,
        )

        return {
            "competition_id": result.competition_id,
            "ranking": result.ranking,
            "my_rank": my_rank,
            "reward": result.rewards.get(self.holon_id, 0),
            "scores": result.scores.get(self.holon_id, {}),
        }

    async def update_reputation(
        self,
        event_type: str,
        outcome: str,
        rating: float = 0.5,
    ) -> None:
        """更新自己的声誉.

        Args:
            event_type: 事件类型 (task, collaboration, competition)
            outcome: 结果 (success, failure)
            rating: 评分 0-1
        """
        from holonpolis.services.market_service import MarketService

        market = MarketService()
        reputation = market._get_reputation(self.holon_id)

        reputation.update(event_type, outcome, rating)
        market.persist_state()

        logger.debug(
            "reputation_updated",
            holon_id=self.holon_id,
            event=event_type,
            outcome=outcome,
            new_score=reputation.overall_score,
        )

    async def find_collaborators(
        self,
        skill_needed: str,
        min_reputation: float = 0.3,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """寻找潜在协作者.

        Args:
            skill_needed: 需要的技能
            min_reputation: 最低声誉要求
            top_k: 返回数量

        Returns:
            List of potential collaborators
        """
        from holonpolis.services.collaboration_service import CollaborationService

        service = CollaborationService()

        candidates = await service.find_collaborators(
            holon_id=self.holon_id,
            skill_needed=skill_needed,
            min_reputation=min_reputation,
            top_k=top_k,
        )

        return [
            {"holon_id": hid, "compatibility_score": score}
            for hid, score in candidates
        ]
