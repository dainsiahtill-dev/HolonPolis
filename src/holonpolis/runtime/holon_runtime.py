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
from collections import Counter
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
from holonpolis.services.holon_service import HolonService, HolonUnavailableError
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
    origin: str = "manual"
    attempt_index: int = 1
    lineage_id: Optional[str] = None
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
        self._holon_service = HolonService()
        self.blueprint = blueprint or self._load_blueprint()
        self.memory = MemoryService(holon_id)
        self.memory.factory.init_holon_tables(holon_id)
        self._reusable_code_library = None
        self._ui_component_library = None
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
        return self._holon_service.get_blueprint(self.holon_id)

    def _has_inflight_evolution(self) -> bool:
        """Return True when any local evolution request is still non-terminal."""
        terminal = {
            EvolutionStatus.COMPLETED,
            EvolutionStatus.FAILED,
            EvolutionStatus.REJECTED,
        }
        return any(request.status not in terminal for request in self._evolution_requests.values())

    def _assert_runtime_available(self, *, action: str) -> None:
        """Block runtime work while the Holon is pending or frozen."""
        if self._has_inflight_evolution():
            raise HolonUnavailableError(holon_id=self.holon_id, status="pending", action=action)
        self._holon_service.assert_runnable(self.holon_id, action=action)

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

    @staticmethod
    def _summarize_reflection_memory_hit(content: Any, limit: int = 220) -> str:
        """Convert stored reflection memory into a concise prompt-ready line."""
        text = str(content or "").strip()
        prefix = "SELF_REFLECTION "
        if text.startswith(prefix):
            raw_payload = text[len(prefix):].strip()
            try:
                payload = json.loads(raw_payload)
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                summary = str(payload.get("summary") or "").strip()
                gaps = payload.get("gaps")
                gap_items = [
                    str(item).strip()
                    for item in (gaps if isinstance(gaps, list) else [])
                    if str(item).strip()
                ]
                suggestions = payload.get("suggestions")
                suggestion_items = [
                    str(item).strip()
                    for item in (suggestions if isinstance(suggestions, list) else [])
                    if str(item).strip()
                ]
                details: List[str] = []
                if summary:
                    details.append(summary)
                if gap_items:
                    details.append(f"gaps={', '.join(gap_items[:3])}")
                if suggestion_items:
                    details.append(f"suggestions={', '.join(suggestion_items[:2])}")
                if details:
                    return " | ".join(details)
        normalized = re.sub(r"\s+", " ", text)
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip() + "..."

    async def _build_reflection_context(
        self,
        user_message: str,
    ) -> str:
        """Build a high-priority self-reflection context block for chat prompts."""
        lines: List[str] = []
        snapshot = self.get_self_reflection(history_limit=2)
        if str(snapshot.get("status") or "").strip().lower() != "no_reflection":
            summary = str(snapshot.get("summary") or "").strip()
            if summary:
                lines.append(f"- Latest reflection: {summary}")

            capability_gaps = snapshot.get("capability_gaps")
            if isinstance(capability_gaps, list):
                gap_ids = [
                    str(item.get("gap_id") or "").strip()
                    for item in capability_gaps
                    if isinstance(item, dict) and str(item.get("gap_id") or "").strip()
                ]
                if gap_ids:
                    lines.append(f"- Active gaps: {', '.join(gap_ids[:3])}")

            auto_evolution = snapshot.get("auto_evolution")
            if isinstance(auto_evolution, dict) and bool(auto_evolution.get("triggered")):
                requests = auto_evolution.get("requests")
                if isinstance(requests, list) and requests:
                    first = requests[0] if isinstance(requests[0], dict) else {}
                    if isinstance(first, dict):
                        skill_name = str(first.get("skill_name") or "").strip()
                        request_id = str(first.get("request_id") or "").strip()
                        if skill_name or request_id:
                            lines.append(
                                "- In-flight self-evolution: "
                                f"{skill_name or 'unknown_skill'} ({request_id or 'unknown_request'})"
                            )

        try:
            reflection_hits = await self.memory.hybrid_search(
                query=user_message,
                top_k=2,
                vector_weight=0.5,
                text_weight=0.5,
                filters={"tags": ["self_reflection"]},
                min_score=0.0,
            )
        except Exception as exc:
            logger.warning(
                "self_reflection_context_search_failed",
                holon_id=self.holon_id,
                error=str(exc),
            )
            reflection_hits = []

        if reflection_hits:
            seen_hit_lines: set[str] = set()
            for item in reflection_hits:
                rendered = self._summarize_reflection_memory_hit(item.content)
                normalized = rendered.strip()
                if not normalized or normalized in seen_hit_lines:
                    continue
                seen_hit_lines.add(normalized)
                lines.append(f"- Retrieved lesson: {normalized}")

        if not lines:
            return ""
        return "\n\n# Self Reflection Guidance\n" + "\n".join(lines)

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
        self._assert_runtime_available(action="chat")
        start_time = time.time()

        # 1. Recall relevant memories
        memories = await self.memory.recall(
            query=user_message,
            top_k=5,
        )
        non_ui_memories = [
            memory
            for memory in memories
            if "ui-component-library" not in set(memory.get("tags", []))
            and "reusable-code-library" not in set(memory.get("tags", []))
        ]

        # 2. Build LLM messages
        system_prompt = self._build_system_prompt()
        reflection_context = await self._build_reflection_context(user_message)
        reusable_code_context = await self._build_reusable_code_context(user_message)
        ui_component_context = await self._build_ui_component_context(user_message)

        # Add memory context if available
        memory_sections: List[str] = []
        if reflection_context:
            memory_sections.append(reflection_context)
        if reusable_code_context:
            memory_sections.append(reusable_code_context)
        if ui_component_context:
            memory_sections.append(ui_component_context)

        if non_ui_memories:
            memory_sections.append(
                "\n\n# Relevant Memories\n" + "\n".join(
                    f"- [{m['kind']}] {m['content'][:200]}"
                    for m in non_ui_memories
                )
            )

        memory_context = "".join(memory_sections)

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
                "memories_recalled": len(non_ui_memories),
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

    def _get_ui_component_library_service(self):
        """Lazy-load the UI component library service."""
        if self._ui_component_library is None:
            from holonpolis.services.ui_component_library_service import UIComponentLibraryService

            self._ui_component_library = UIComponentLibraryService(
                self.holon_id,
                memory_service=self.memory,
            )
        return self._ui_component_library

    def _get_reusable_code_library_service(self):
        """Lazy-load the reusable code asset library service."""
        if self._reusable_code_library is None:
            from holonpolis.services.reusable_code_library_service import ReusableCodeLibraryService

            self._reusable_code_library = ReusableCodeLibraryService(
                self.holon_id,
                memory_service=self.memory,
            )
        return self._reusable_code_library

    async def index_reusable_code_library(
        self,
        source_path: str,
        *,
        library_name: str,
        library_kind: str = "code_asset",
        framework: str = "generic",
        store_mode: str = "full",
        include_extensions: Optional[List[str]] = None,
        max_file_bytes: int = 60000,
    ) -> Dict[str, Any]:
        """Index a reusable code library into per-Holon memory."""
        self._assert_runtime_available(action="index_code_library")
        self.enforce_capability(
            "code.library.index",
            aliases=["code_library", "index"],
        )
        service = self._get_reusable_code_library_service()
        return await service.index_local_library(
            source_path=source_path,
            library_name=library_name,
            library_kind=library_kind,
            framework=framework,
            store_mode=store_mode,
            include_extensions=include_extensions,
            max_file_bytes=max_file_bytes,
        )

    async def search_reusable_code_library(
        self,
        query: str,
        *,
        top_k: int = 3,
        library_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search reusable code assets stored in memory."""
        self.enforce_capability(
            "code.library.search",
            aliases=["code_library", "search", "retrieve"],
        )
        service = self._get_reusable_code_library_service()
        return await service.search_assets(
            query=query,
            top_k=top_k,
            library_kind=library_kind,
        )

    async def index_ui_component_library(
        self,
        source_path: str,
        *,
        library_name: str,
        framework: str = "react",
        store_mode: str = "full",
        include_extensions: Optional[List[str]] = None,
        max_file_bytes: int = 60000,
    ) -> Dict[str, Any]:
        """Index a local UI component library into per-Holon memory."""
        self._assert_runtime_available(action="index_ui_library")
        self.enforce_capability(
            "ui.library.index",
            aliases=["ui_library", "index"],
        )
        service = self._get_ui_component_library_service()
        return await service.index_local_library(
            source_path=source_path,
            library_name=library_name,
            framework=framework,
            store_mode=store_mode,
            include_extensions=include_extensions,
            max_file_bytes=max_file_bytes,
        )

    async def search_ui_component_library(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search the indexed UI component memory for reusable code."""
        self.enforce_capability(
            "ui.library.search",
            aliases=["ui_library", "search", "retrieve"],
        )
        service = self._get_ui_component_library_service()
        return await service.search_components(query=query, top_k=top_k)

    @staticmethod
    def _looks_like_code_generation_request(user_message: str) -> bool:
        """Heuristically detect project/code generation requests."""
        normalized = str(user_message or "").strip().lower()
        if not normalized:
            return False

        keywords = (
            "build",
            "create",
            "generate",
            "implement",
            "scaffold",
            "starter",
            "template",
            "sdk",
            "client",
            "api",
            "backend",
            "frontend",
            "library",
            "package",
            "project",
            "代码",
            "项目",
            "生成",
            "构建",
            "搭建",
            "实现",
            "脚手架",
            "组件库",
            "代码库",
        )
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _looks_like_frontend_request(user_message: str) -> bool:
        """Heuristically detect frontend/UI-oriented requests."""
        normalized = str(user_message or "").strip().lower()
        if not normalized:
            return False

        keywords = (
            "ui",
            "ux",
            "frontend",
            "front-end",
            "component",
            "react",
            "vue",
            "svelte",
            "tailwind",
            "page",
            "layout",
            "button",
            "form",
            "dashboard",
            "design system",
            "css",
            "style",
            "theme",
            "front end",
            "前端",
            "组件",
            "页面",
            "界面",
            "按钮",
            "表单",
            "布局",
            "样式",
            "导航",
        )
        return any(keyword in normalized for keyword in keywords)

    async def _build_ui_component_context(self, user_message: str) -> str:
        """Build a prompt context block with retrieved UI component code."""
        if not self._looks_like_frontend_request(user_message):
            return ""
        if not self.is_capability_allowed(
            "ui.library.search",
            aliases=["ui_library", "search", "retrieve"],
        ):
            return ""

        service = self._get_ui_component_library_service()
        try:
            return await service.build_prompt_context(
                user_message,
                top_k=2,
                max_code_chars=2600,
            )
        except Exception as exc:
            logger.warning(
                "ui_component_context_build_failed",
                holon_id=self.holon_id,
                error=str(exc),
            )
            return ""

    async def _build_reusable_code_context(self, user_message: str) -> str:
        """Build a prompt context block with retrieved reusable code assets."""
        if not self._looks_like_code_generation_request(user_message):
            return ""
        if not self.is_capability_allowed(
            "code.library.search",
            aliases=["code_library", "search", "retrieve"],
        ):
            return ""

        service = self._get_reusable_code_library_service()
        try:
            return await service.build_prompt_context(
                user_message,
                top_k=2,
                max_code_chars=2600,
                library_kind="code_asset",
                heading="Retrieved Reusable Code Assets",
            )
        except Exception as exc:
            logger.warning(
                "reusable_code_context_build_failed",
                holon_id=self.holon_id,
                error=str(exc),
            )
            return ""

    async def request_evolution(
        self,
        skill_name: str,
        description: str,
        requirements: List[str],
        test_cases: Optional[List[Dict[str, Any]]] = None,
        parent_skills: Optional[List[str]] = None,
        *,
        origin: str = "manual",
        attempt_index: int = 1,
        lineage_id: Optional[str] = None,
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
        self._assert_runtime_available(action="evolve")
        self.enforce_capability("evolution.request", aliases=["evolve", "execute"])
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
            origin=str(origin or "manual").strip() or "manual",
            attempt_index=max(1, int(attempt_index)),
            lineage_id=str(lineage_id or "").strip() or None,
        )

        # Store request
        self.state.evolution_requests.append(request_id)
        self._evolution_requests[request_id] = request
        self._holon_service.mark_pending(
            self.holon_id,
            reason="evolution_requested",
            details={"service": "holon_runtime", "request_id": request_id},
        )
        try:
            await self.remember(
                content=f"Evolution request {request_id}: {skill_name} - {description}",
                tags=["evolution", "skill_request"],
                importance=0.9,
            )
        except Exception:
            self.state.evolution_requests.remove(request_id)
            self._evolution_requests.pop(request_id, None)
            if self._holon_service.get_holon_status(self.holon_id) == "pending":
                self._holon_service.mark_active(
                    self.holon_id,
                    reason="evolution_request_aborted",
                    details={"service": "holon_runtime", "request_id": request_id},
                )
            raise

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
                    pending_token=request.request_id,
                )
            else:
                # Autonomous mode: LLM generates tests + code inside EvolutionService.
                result = await evolution_service.evolve_skill_autonomous(
                    holon_id=self.holon_id,
                    skill_name=request.skill_name,
                    description=request.description,
                    requirements=request.requirements,
                    tool_schema=tool_schema,
                    pending_token=request.request_id,
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
            self._finalize_evolution_audit(
                request,
                status=EvolutionStatus.FAILED,
                phase="runtime",
                error_message=request.error_message,
            )
            if self._holon_service.get_holon_status(self.holon_id) == "pending":
                self._holon_service.mark_active(
                    self.holon_id,
                    reason="evolution_failed_early",
                    details={"service": "holon_runtime", "request_id": request.request_id},
                )
            return

        if result and result.success:
            request.status = EvolutionStatus.COMPLETED
            request.result = {
                "skill_id": result.skill_id,
                "attestation_id": result.attestation.attestation_id if result.attestation else None,
                "code_path": result.code_path,
                "test_path": getattr(result, "test_path", None),
                "manifest_path": getattr(result, "manifest_path", None),
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
            self._finalize_evolution_audit(
                request,
                status=EvolutionStatus.COMPLETED,
                phase="complete",
                result_payload=request.result,
            )
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
        self._finalize_evolution_audit(
            request,
            status=EvolutionStatus.FAILED,
            phase=str(result.phase if result else "runtime").strip() or "runtime",
            error_message=request.error_message,
        )

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

    @staticmethod
    def _build_reflection_memory_content(reflection: Dict[str, Any]) -> str:
        """Render a compact structured summary suitable for retrieval."""
        payload = reflection if isinstance(reflection, dict) else {}
        trigger = payload.get("trigger")
        trigger_map = trigger if isinstance(trigger, dict) else {}
        capability_gaps = payload.get("capability_gaps")
        gap_list = capability_gaps if isinstance(capability_gaps, list) else []
        suggestions = payload.get("suggestions")
        suggestion_list = suggestions if isinstance(suggestions, list) else []
        auto_evolution = payload.get("auto_evolution")
        auto_map = auto_evolution if isinstance(auto_evolution, dict) else {}

        gap_ids = [
            str(item.get("gap_id") or "").strip()
            for item in gap_list
            if isinstance(item, dict) and str(item.get("gap_id") or "").strip()
        ]
        suggestion_tokens: List[str] = []
        for item in suggestion_list:
            if not isinstance(item, dict):
                continue
            token = str(
                item.get("suggested_skill")
                or item.get("skill_name")
                or item.get("type")
                or ""
            ).strip()
            if token:
                suggestion_tokens.append(token)
            if len(suggestion_tokens) >= 3:
                break

        envelope = {
            "reflection_id": str(payload.get("reflection_id") or "").strip(),
            "status": str(payload.get("status") or "").strip(),
            "summary": str(payload.get("summary") or "").strip(),
            "trigger_type": str(trigger_map.get("type") or "").strip(),
            "trigger_request_id": str(trigger_map.get("request_id") or "").strip(),
            "gaps": gap_ids[:4],
            "suggestions": suggestion_tokens,
            "auto_evolution": {
                "triggered": bool(auto_map.get("triggered")),
                "request_count": int(auto_map.get("request_count", 0) or 0),
            },
        }
        return "SELF_REFLECTION " + json.dumps(envelope, sort_keys=True)

    async def _safe_persist_reflection_memory(
        self,
        reflection: Dict[str, Any],
        *,
        phase_tag: str,
    ) -> None:
        """Persist structured self-reflection as retrievable memory."""
        payload = reflection if isinstance(reflection, dict) else {}
        phase = str(phase_tag or "analysis").strip().lower() or "analysis"
        tags = ["self_reflection", phase]
        status = str(payload.get("status") or "").strip().lower()
        if status:
            tags.append(status)

        capability_gaps = payload.get("capability_gaps")
        if isinstance(capability_gaps, list):
            for item in capability_gaps:
                if not isinstance(item, dict):
                    continue
                gap_id = str(item.get("gap_id") or "").strip().lower()
                if gap_id:
                    tags.append(gap_id)
                if len(tags) >= 6:
                    break

        deduped_tags: List[str] = []
        seen_tags: set[str] = set()
        for tag in tags:
            normalized = str(tag or "").strip().lower()
            if not normalized or normalized in seen_tags:
                continue
            seen_tags.add(normalized)
            deduped_tags.append(normalized)

        try:
            await self.memory.remember(
                content=self._build_reflection_memory_content(payload),
                kind=MemoryKind.PATTERN,
                tags=deduped_tags,
                importance=0.95 if phase == "evolution_failure" else 0.85,
            )
        except Exception as exc:
            logger.warning(
                "self_reflection_memory_write_failed",
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

    def _max_reflective_attempts_for_request(self) -> int:
        """Cap autonomous cross-request retries using the blueprint policy."""
        configured = max(1, int(self.blueprint.evolution_policy.max_evolution_attempts))
        strategy = self.blueprint.evolution_policy.strategy
        if strategy.value == "conservative":
            return 1
        if strategy.value == "balanced":
            return min(configured, 2)
        return configured

    @staticmethod
    def _summarize_repair_actions(actions: Any, limit: int = 2) -> List[str]:
        """Keep follow-up repair actions compact and deterministic."""
        if not isinstance(actions, list):
            return []
        normalized: List[str] = []
        for item in actions:
            text = str(item or "").strip()
            if not text:
                continue
            normalized.append(text)
            if len(normalized) >= max(1, int(limit)):
                break
        return normalized

    async def _persist_failure_reflection_snapshot(
        self,
        request: EvolutionRequest,
        improvement_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist a targeted post-mortem snapshot so failures become reusable context."""
        next_round = improvement_plan.get("next_round_plan")
        next_round_plan = next_round if isinstance(next_round, dict) else {}
        reflection_id = f"reflect_{uuid.uuid4().hex[:12]}"
        created_at = utc_now_iso()
        failure_phase = str(improvement_plan.get("failure_phase") or "unknown").strip()
        failure_summary = self._normalize_reflection_message(
            improvement_plan.get("failure_summary") or request.error_message or "evolution_failed"
        )
        max_attempts = self._max_reflective_attempts_for_request()
        remaining_attempts = max(0, max_attempts - int(request.attempt_index))
        can_continue = bool(
            improvement_plan.get("retry_recommended")
            and remaining_attempts > 0
        )

        capability_gaps = [
            {
                "gap_id": str(next_round_plan.get("focus") or "evolution_resilience").strip() or "evolution_resilience",
                "severity": "high",
                "reason": failure_summary,
            }
        ]
        summary = (
            f"Evolution reflection for {request.skill_name}: "
            f"phase={failure_phase}, attempt={request.attempt_index}/{max_attempts}, "
            f"continuation={'yes' if can_continue else 'no'}"
        )
        snapshot: Dict[str, Any] = {
            "status": "evolution_failure_reflected",
            "reflection_id": reflection_id,
            "created_at": created_at,
            "summary": summary,
            "trigger": {
                "type": "evolution_failure",
                "request_id": request.request_id,
                "lineage_id": request.lineage_id or request.request_id,
            },
            "metrics": {
                "failure_phase": failure_phase,
                "attempt_index": int(request.attempt_index),
                "max_reflective_attempts": max_attempts,
                "remaining_reflective_attempts": remaining_attempts,
                "continuation_allowed": can_continue,
            },
            "capability_gaps": capability_gaps,
            "suggestions": [
                {
                    "type": "retry_same_skill",
                    "reason": failure_summary,
                    "skill_name": request.skill_name,
                    "repair_actions": self._summarize_repair_actions(
                        next_round_plan.get("repair_actions", [])
                    ),
                    "revised_requirements": list(next_round_plan.get("revised_requirements", []))
                    if isinstance(next_round_plan.get("revised_requirements"), list)
                    else [],
                }
            ],
            "auto_evolution": {
                "requested": can_continue,
                "triggered": False,
                "request_count": 0,
                "requests": [],
                "skipped": [],
            },
        }

        self._holon_service.record_self_reflection(self.holon_id, snapshot)
        await self._safe_remember(
            content=summary,
            tags=["self_improvement", "reflection", "evolution_failure", request.skill_name],
            importance=0.95,
        )
        return snapshot

    async def _continue_evolution_after_failure(
        self,
        request: EvolutionRequest,
        improvement_plan: Dict[str, Any],
    ) -> Optional[EvolutionRequest]:
        """Schedule one follow-up evolution request when policy still allows it."""
        max_attempts = self._max_reflective_attempts_for_request()
        if int(request.attempt_index) >= max_attempts:
            return None

        next_round = improvement_plan.get("next_round_plan")
        next_round_plan = next_round if isinstance(next_round, dict) else {}
        revised_requirements = next_round_plan.get("revised_requirements")
        if not isinstance(revised_requirements, list) or not revised_requirements:
            return None

        follow_up_description = (
            f"{request.description} | self-repair attempt {int(request.attempt_index) + 1}"
        ).strip()
        follow_up = await self.request_evolution(
            skill_name=request.skill_name,
            description=follow_up_description,
            requirements=[str(item).strip() for item in revised_requirements if str(item).strip()],
            test_cases=list(request.test_cases),
            parent_skills=list(request.parent_skills),
            origin="self_reflection_recovery",
            attempt_index=int(request.attempt_index) + 1,
            lineage_id=request.lineage_id or request.request_id,
        )
        return follow_up

    @staticmethod
    def _summarize_failure_message(message: Any, limit: int = 220) -> str:
        """Compact noisy failure text for telemetry and next-step planning."""
        text = re.sub(r"\s+", " ", str(message or "").strip())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _build_failure_improvement_plan(
        self,
        request: EvolutionRequest,
        result: Any,
    ) -> Dict[str, Any]:
        """Turn an evolution failure into a concrete next-round improvement plan."""
        phase = str(getattr(result, "phase", "") or "unknown").strip().lower()
        failure_summary = self._summarize_failure_message(
            getattr(result, "error_message", "") or request.error_message or "evolution_failed"
        )

        focus_map: Dict[str, Dict[str, Any]] = {
            "red": {
                "focus": "tighten_contract",
                "actions": [
                    "Narrow the interface to one deterministic entrypoint with explicit parameters.",
                    "Regenerate tests so they validate the contract without hidden assumptions.",
                    "Prefer simpler test fixtures and fewer branches in the first repair round.",
                ],
            },
            "green": {
                "focus": "stabilize_execution",
                "actions": [
                    "Reduce logic complexity and keep the implementation pure and deterministic.",
                    "Handle empty or malformed input without raising exceptions.",
                    "Return the exact schema expected by tests before adding optimizations.",
                ],
            },
            "verify": {
                "focus": "remove_restricted_constructs",
                "actions": [
                    "Remove banned imports, dynamic execution, and reflective attribute access.",
                    "Use only safe standard-language constructs that pass AST verification.",
                    "Re-check helper utilities for hidden unsafe calls before retrying.",
                ],
            },
            "persist": {
                "focus": "stabilize_artifacts",
                "actions": [
                    "Keep output paths relative and within the Holon sandbox.",
                    "Ensure manifest and attestation data are complete and serializable.",
                    "Avoid emitting malformed metadata in the next retry.",
                ],
            },
        }
        defaults = {
            "focus": "reduce_scope",
            "actions": [
                "Shrink the requirement surface and retry with the smallest useful behavior.",
                "Preserve deterministic behavior first, then add complexity in later iterations.",
                "Use the latest failure as a hard constraint for the next prompt.",
            ],
        }
        chosen = focus_map.get(phase, defaults)
        revised_requirements = list(request.requirements)
        revised_requirements.append(f"Address previous {phase} failure: {failure_summary}")

        retry_prompt = (
            f"Previous evolution for skill '{request.skill_name}' failed in phase '{phase}'. "
            f"Failure summary: {failure_summary}. "
            f"Next-round focus: {chosen['focus']}. "
            "Follow the listed repair actions and keep the implementation minimal, deterministic, and sandbox-safe."
        )

        return {
            "status": "failed",
            "request_id": request.request_id,
            "skill_name": request.skill_name,
            "failure_phase": phase,
            "failure_summary": failure_summary,
            "retry_recommended": True,
            "next_round_plan": {
                "focus": chosen["focus"],
                "repair_actions": list(chosen["actions"]),
                "revised_requirements": revised_requirements,
                "retry_prompt": retry_prompt,
            },
        }

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

        improvement_plan = self._build_failure_improvement_plan(request, result)
        request.result = improvement_plan

        reflection_snapshot = await self._persist_failure_reflection_snapshot(request, improvement_plan)

        await self.remember(
            content=(
                f"Evolution failed for {request.skill_name}: {improvement_plan['failure_summary']} | "
                f"next_focus={improvement_plan['next_round_plan']['focus']}"
            ),
            tags=["evolution", "failure", "improvement_plan", request.skill_name],
            importance=0.8,
        )

        follow_up: Optional[EvolutionRequest] = None
        try:
            follow_up = await self._continue_evolution_after_failure(request, improvement_plan)
        except Exception as exc:
            logger.warning(
                "evolution_failure_continuation_failed",
                holon_id=self.holon_id,
                request_id=request.request_id,
                error=str(exc),
            )
            improvement_plan["follow_up"] = {
                "triggered": False,
                "error": str(exc),
            }
            reflection_snapshot["auto_evolution"] = {
                "requested": True,
                "triggered": False,
                "request_count": 0,
                "requests": [],
                "skipped": [{"reason": "request_failed", "detail": str(exc)}],
            }
            self._holon_service.record_self_reflection(self.holon_id, reflection_snapshot)
            await self._safe_persist_reflection_memory(
                reflection_snapshot,
                phase_tag="evolution_failure",
            )
            return

        if follow_up is None:
            improvement_plan["follow_up"] = {
                "triggered": False,
                "reason": "budget_exhausted_or_missing_revised_requirements",
            }
            reflection_snapshot["auto_evolution"] = {
                "requested": False,
                "triggered": False,
                "request_count": 0,
                "requests": [],
                "skipped": [
                    {
                        "reason": "budget_exhausted_or_missing_revised_requirements",
                    }
                ],
            }
            self._holon_service.record_self_reflection(self.holon_id, reflection_snapshot)
            await self._safe_persist_reflection_memory(
                reflection_snapshot,
                phase_tag="evolution_failure",
            )
            return

        improvement_plan["follow_up"] = {
            "triggered": True,
            "request_id": follow_up.request_id,
            "attempt_index": follow_up.attempt_index,
            "lineage_id": follow_up.lineage_id,
            "origin": follow_up.origin,
        }
        reflection_snapshot["auto_evolution"] = {
            "requested": True,
            "triggered": True,
            "request_count": 1,
            "requests": [
                {
                    "request_id": follow_up.request_id,
                    "skill_name": follow_up.skill_name,
                    "attempt_index": follow_up.attempt_index,
                    "lineage_id": follow_up.lineage_id,
                    "origin": follow_up.origin,
                }
            ],
            "skipped": [],
        }
        self._holon_service.record_self_reflection(self.holon_id, reflection_snapshot)
        await self._safe_persist_reflection_memory(
            reflection_snapshot,
            phase_tag="evolution_failure",
        )

    @staticmethod
    def _normalize_reflection_message(value: Any) -> str:
        """Normalize noisy error text into a compact reflection signal."""
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text or "unknown"

    @classmethod
    def _classify_failure_category(
        cls,
        detail: Any,
        *,
        phase: str = "",
    ) -> str:
        """Bucket failures into reusable adaptation categories."""
        normalized_phase = str(phase or "").strip().lower()
        if normalized_phase in {"red", "green", "verify", "persist"}:
            return f"evolution_{normalized_phase}"

        text = cls._normalize_reflection_message(detail).lower()
        if any(
            token in text
            for token in (
                "schema",
                "payload",
                "contract",
                "validation",
                "missing required",
                "unexpected field",
                "missing required field",
            )
        ):
            return "contract"
        if any(
            token in text
            for token in ("memory", "recall", "retrieval", "context window", "context")
        ):
            return "memory"
        if any(
            token in text
            for token in (
                "security",
                "unsafe",
                "forbidden",
                "denied",
                "static scan",
                "ast",
                "sandbox",
            )
        ):
            return "safety"
        if any(
            token in text
            for token in ("timeout", "timed out", "latency", "slow", "deadline", "too long")
        ):
            return "latency"
        if any(
            token in text
            for token in ("tool", "subprocess", "permission", "filesystem", "write access")
        ):
            return "tooling"
        return "execution"

    def _collect_failure_signals(
        self,
        recent_episodes: List[Dict[str, Any]],
        latest_audit: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Collect failure signals from episodes plus recent evolution telemetry."""
        signals: List[Dict[str, str]] = []

        for episode in recent_episodes:
            if str(episode.get("outcome") or "").strip().lower() != "failure":
                continue
            details = episode.get("outcome_details")
            detail_map = details if isinstance(details, dict) else {}
            message = (
                detail_map.get("error")
                or detail_map.get("message")
                or detail_map.get("reason")
                or "episode_failed"
            )
            normalized_message = self._normalize_reflection_message(message)
            signals.append(
                {
                    "source": "episode",
                    "category": self._classify_failure_category(normalized_message),
                    "message": normalized_message,
                }
            )

        for request in self._evolution_requests.values():
            if request.status != EvolutionStatus.FAILED:
                continue
            result_payload = request.result if isinstance(request.result, dict) else {}
            failure_phase = ""
            if isinstance(result_payload, dict):
                failure_phase = str(result_payload.get("failure_phase") or "").strip()
            message = (
                request.error_message
                or (result_payload.get("failure_summary") if isinstance(result_payload, dict) else "")
                or "evolution_failed"
            )
            normalized_message = self._normalize_reflection_message(message)
            signals.append(
                {
                    "source": "evolution_request",
                    "category": self._classify_failure_category(
                        normalized_message,
                        phase=failure_phase,
                    ),
                    "message": normalized_message,
                }
            )

        audit_result = str(latest_audit.get("result") or "").strip().lower()
        audit_error = str(latest_audit.get("error") or "").strip()
        if audit_result == "failed" or audit_error:
            audit_phase = str(latest_audit.get("phase") or "").strip()
            normalized_message = self._normalize_reflection_message(audit_error or "evolution_audit_failed")
            signals.append(
                {
                    "source": "state_audit",
                    "category": self._classify_failure_category(
                        normalized_message,
                        phase=audit_phase,
                    ),
                    "message": normalized_message,
                }
            )

        return signals

    def _derive_self_improvement_actions(
        self,
        *,
        total_episodes: int,
        success_rate: float,
        failures: int,
        avg_latency_ms: int,
        skill_count: int,
        failure_categories: Dict[str, int],
        max_suggestions: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Convert reflection metrics into concrete capability gaps and repair actions."""
        capability_gaps: List[Dict[str, Any]] = []
        suggestions: List[Dict[str, Any]] = []
        seen_gaps: set[str] = set()
        seen_skills: set[str] = set()

        def add_gap(gap_id: str, severity: str, reason: str) -> None:
            if gap_id in seen_gaps:
                return
            seen_gaps.add(gap_id)
            capability_gaps.append(
                {
                    "gap_id": gap_id,
                    "severity": severity,
                    "reason": reason,
                }
            )

        def add_evolution_suggestion(
            *,
            priority: str,
            suggested_skill: str,
            reason: str,
            description: str,
            requirements: List[str],
        ) -> None:
            if len(suggestions) >= max_suggestions:
                return
            normalized_skill = str(suggested_skill or "").strip().lower()
            if not normalized_skill or normalized_skill in seen_skills:
                return
            seen_skills.add(normalized_skill)
            suggestions.append(
                {
                    "type": "evolve_skill",
                    "priority": priority,
                    "reason": reason,
                    "suggested_skill": suggested_skill,
                    "description": description,
                    "requirements": list(requirements),
                }
            )

        def add_memory_suggestion(priority: str, reason: str, action: str) -> None:
            if len(suggestions) >= max_suggestions:
                return
            suggestions.append(
                {
                    "type": "improve_memory",
                    "priority": priority,
                    "reason": reason,
                    "action": action,
                }
            )

        contract_failures = int(failure_categories.get("contract", 0) or 0) + int(
            failure_categories.get("evolution_red", 0) or 0
        )
        execution_failures = int(failure_categories.get("execution", 0) or 0) + int(
            failure_categories.get("evolution_green", 0) or 0
        )
        safety_failures = int(failure_categories.get("safety", 0) or 0) + int(
            failure_categories.get("evolution_verify", 0) or 0
        )
        memory_failures = int(failure_categories.get("memory", 0) or 0)
        latency_failures = int(failure_categories.get("latency", 0) or 0)

        if contract_failures > 0:
            add_gap(
                "input_contracts",
                "high",
                f"Detected {contract_failures} contract/schema failures in recent behavior.",
            )
            add_evolution_suggestion(
                priority="high",
                suggested_skill="input_contract_guard",
                reason="Repeated contract mismatches indicate unstable external interfaces.",
                description="Validate and normalize inbound payloads before core execution.",
                requirements=[
                    "Reject malformed payloads deterministically.",
                    "Return machine-readable validation errors.",
                    "Preserve sandbox-safe pure functions only.",
                ],
            )

        if execution_failures > 0 or (total_episodes > 0 and success_rate < 0.8):
            add_gap(
                "execution_resilience",
                "high" if execution_failures > 0 else "medium",
                (
                    f"Success rate is {success_rate:.1%} with {execution_failures} execution-side failures."
                    if total_episodes > 0
                    else "Execution resilience is unproven."
                ),
            )
            add_evolution_suggestion(
                priority="high" if execution_failures > 0 else "medium",
                suggested_skill="reliability_guard",
                reason="Core execution paths need deterministic guardrails and safer fallbacks.",
                description="Harden runtime execution with normalization, guard clauses, and deterministic fallbacks.",
                requirements=[
                    "Handle empty and malformed input without raising exceptions.",
                    "Keep output deterministic for identical inputs.",
                    "Prefer small pure helper functions over branching side effects.",
                ],
            )

        if safety_failures > 0:
            add_gap(
                "sandbox_safety",
                "high",
                f"Detected {safety_failures} safety or verification failures in evolution output.",
            )
            add_evolution_suggestion(
                priority="high",
                suggested_skill="sandbox_compliance_guard",
                reason="Evolution retries are leaking unsafe constructs into generated code.",
                description="Pre-screen generated code for unsafe imports and sandbox-breaking patterns.",
                requirements=[
                    "Reject banned imports and dynamic execution constructs.",
                    "Surface concise remediation guidance for verify-phase failures.",
                    "Keep all checks deterministic and offline.",
                ],
            )

        if memory_failures > 0 or failures >= max(3, total_episodes // 3 if total_episodes > 0 else 3):
            add_gap(
                "memory_retrieval",
                "medium",
                "Recent failures suggest the Holon is not reusing prior lessons effectively.",
            )
            add_memory_suggestion(
                priority="medium",
                reason="Promote more structured postmortems so failed episodes become retrievable lessons.",
                action="promote_failure_postmortems",
            )

        if latency_failures > 0 or avg_latency_ms >= 2500:
            add_gap(
                "latency_control",
                "medium",
                f"Average latency is {avg_latency_ms}ms with {latency_failures} latency-related failures.",
            )
            add_evolution_suggestion(
                priority="medium",
                suggested_skill="latency_optimizer",
                reason="Latency is becoming a usability bottleneck.",
                description="Trim avoidable runtime overhead and keep responses within predictable latency budgets.",
                requirements=[
                    "Prefer early exits for invalid work.",
                    "Avoid redundant LLM or tool calls for repeated requests.",
                    "Emit deterministic metrics suitable for regression tests.",
                ],
            )

        if skill_count < 2:
            add_gap(
                "capability_divergence",
                "medium",
                "The Holon has too few differentiated skills to explore adjacent problem spaces.",
            )
            add_evolution_suggestion(
                priority="medium",
                suggested_skill="capability_explorer",
                reason="Low skill diversity limits adaptive range and long-term usefulness.",
                description="Generate narrow adjacent capabilities that expand the Holon's usable frontier.",
                requirements=[
                    "Propose one adjacent capability at a time.",
                    "Keep each new capability testable through a deterministic execute contract.",
                    "Avoid duplicating already persisted skills.",
                ],
            )

        if not suggestions:
            add_evolution_suggestion(
                priority="low",
                suggested_skill="adjacent_capability_probe",
                reason="Current telemetry is healthy; expand cautiously into neighboring capabilities.",
                description="Probe one adjacent capability so the Holon keeps diverging instead of stagnating.",
                requirements=[
                    "Keep the scope minimal and deterministic.",
                    "Produce one clearly testable execute entrypoint.",
                    "Prefer composability over broad one-shot behaviors.",
                ],
            )

        return {
            "capability_gaps": capability_gaps,
            "suggestions": suggestions,
        }

    async def _trigger_self_reflective_evolution(
        self,
        suggestions: List[Dict[str, Any]],
        *,
        max_evolution_requests: int,
    ) -> Dict[str, Any]:
        """Best-effort promotion of one reflection result into a new evolution request."""
        summary: Dict[str, Any] = {
            "requested": True,
            "triggered": False,
            "request_count": 0,
            "requests": [],
            "skipped": [],
        }

        request_budget = max(0, int(max_evolution_requests))
        if request_budget <= 0:
            summary["requested"] = False
            return summary

        if request_budget > 1:
            summary["skipped"].append(
                {
                    "reason": "single_inflight_limit",
                    "detail": "Holon runtime serializes self-evolution to one in-flight request.",
                }
            )
        request_budget = min(1, request_budget)

        known_skills = set()
        for skill in self.list_skills():
            skill_id = str(skill.get("skill_id") or "").strip().lower()
            skill_name = str(skill.get("name") or "").strip().lower()
            if skill_id:
                known_skills.add(skill_id)
            if skill_name:
                known_skills.add(skill_name)

        for suggestion in suggestions:
            if summary["request_count"] >= request_budget:
                break

            if str(suggestion.get("type") or "").strip().lower() != "evolve_skill":
                summary["skipped"].append(
                    {
                        "reason": "non_evolution_action",
                        "detail": str(suggestion.get("action") or suggestion.get("type") or "").strip(),
                    }
                )
                continue

            suggested_skill = str(suggestion.get("suggested_skill") or "").strip()
            if not suggested_skill:
                summary["skipped"].append({"reason": "missing_skill_name"})
                continue
            if suggested_skill.lower() in known_skills:
                summary["skipped"].append(
                    {
                        "reason": "skill_already_present",
                        "skill_name": suggested_skill,
                    }
                )
                continue

            try:
                request = await self.request_evolution(
                    skill_name=suggested_skill,
                    description=str(suggestion.get("description") or suggestion.get("reason") or "").strip(),
                    requirements=[
                        str(item).strip()
                        for item in suggestion.get("requirements", [])
                        if str(item).strip()
                    ],
                )
            except CapabilityDeniedError as exc:
                summary["skipped"].append(
                    {
                        "reason": "boundary_denied",
                        "skill_name": suggested_skill,
                        "detail": str(exc),
                    }
                )
                continue
            except Exception as exc:
                logger.warning(
                    "self_reflection_auto_evolution_failed",
                    holon_id=self.holon_id,
                    skill_name=suggested_skill,
                    error=str(exc),
                )
                summary["skipped"].append(
                    {
                        "reason": "request_failed",
                        "skill_name": suggested_skill,
                        "detail": str(exc),
                    }
                )
                continue

            summary["triggered"] = True
            summary["request_count"] = 1
            summary["requests"].append(
                {
                    "request_id": request.request_id,
                    "skill_name": request.skill_name,
                    "status": request.status.value,
                }
            )
            break

        return summary

    async def self_improve(
        self,
        auto_evolve: bool = False,
        max_suggestions: int = 3,
        max_evolution_requests: int = 1,
    ) -> Dict[str, Any]:
        """Run a durable self-reflection pass and optionally spawn one new evolution."""
        self._assert_runtime_available(action="self_improve")

        suggestion_limit = max(1, int(max_suggestions))
        evolution_limit = max(0, int(max_evolution_requests))
        recent_episodes = await self.memory.get_episodes(limit=100)
        persisted_skills = self.list_skills()
        state_payload = self._holon_service.get_holon_state(self.holon_id)
        latest_audit = state_payload.get("evolution_audit")
        if not isinstance(latest_audit, dict):
            latest_audit = {}

        local_requests = list(self._evolution_requests.values())
        if not recent_episodes and not persisted_skills and not local_requests and not latest_audit:
            return {
                "status": "no_data",
                "metrics": {
                    "total_episodes": 0,
                    "success_rate": 0.0,
                    "failure_patterns": {},
                    "failure_categories": {},
                },
                "capability_gaps": [],
                "suggestions": [],
                "auto_evolution": {
                    "requested": bool(auto_evolve and evolution_limit > 0),
                    "triggered": False,
                    "request_count": 0,
                    "requests": [],
                    "skipped": [],
                },
            }

        total = len(recent_episodes)
        successes = sum(1 for episode in recent_episodes if episode.get("outcome") == "success")
        failures = sum(1 for episode in recent_episodes if episode.get("outcome") == "failure")
        success_rate = float(successes / total) if total > 0 else 0.0
        avg_latency_ms = int(
            round(sum(int(episode.get("latency_ms", 0) or 0) for episode in recent_episodes) / total)
        ) if total > 0 else 0
        avg_cost = round(
            sum(float(episode.get("cost", 0.0) or 0.0) for episode in recent_episodes) / total,
            6,
        ) if total > 0 else 0.0

        failure_signals = self._collect_failure_signals(recent_episodes, latest_audit)
        failure_pattern_counter: Counter[str] = Counter(
            signal["message"]
            for signal in failure_signals
            if signal.get("message")
        )
        failure_category_counter: Counter[str] = Counter(
            signal["category"]
            for signal in failure_signals
            if signal.get("category")
        )

        completed_requests = sum(1 for request in local_requests if request.status == EvolutionStatus.COMPLETED)
        failed_requests = sum(1 for request in local_requests if request.status == EvolutionStatus.FAILED)
        total_requests = len(local_requests)
        evolution_success_rate = (
            float(completed_requests / total_requests)
            if total_requests > 0
            else 0.0
        )
        adaptation_pressure = round(
            min(
                1.0,
                (float(failures / total) if total > 0 else 0.0)
                + (0.15 if failed_requests > completed_requests else 0.0)
                + (0.15 if len(persisted_skills) < 2 else 0.0),
            ),
            4,
        )

        action_plan = self._derive_self_improvement_actions(
            total_episodes=total,
            success_rate=success_rate,
            failures=failures,
            avg_latency_ms=avg_latency_ms,
            skill_count=len(persisted_skills),
            failure_categories=dict(failure_category_counter),
            max_suggestions=suggestion_limit,
        )

        summary = (
            f"Self-reflection: success_rate={success_rate:.1%}, "
            f"episodes={total}, skills={len(persisted_skills)}, "
            f"adaptation_pressure={adaptation_pressure:.2f}"
        )
        reflection_id = f"reflect_{uuid.uuid4().hex[:12]}"
        created_at = utc_now_iso()

        reflection: Dict[str, Any] = {
            "status": "analyzed",
            "reflection_id": reflection_id,
            "created_at": created_at,
            "summary": summary,
            "metrics": {
                "total_episodes": total,
                "success_rate": success_rate,
                "failure_patterns": dict(failure_pattern_counter.most_common(5)),
                "failure_categories": dict(failure_category_counter),
                "avg_latency_ms": avg_latency_ms,
                "avg_cost": avg_cost,
                "persisted_skill_count": len(persisted_skills),
                "evolution": {
                    "total_requests": total_requests,
                    "completed": completed_requests,
                    "failed": failed_requests,
                    "success_rate": evolution_success_rate,
                },
                "adaptation_pressure": adaptation_pressure,
            },
            "capability_gaps": action_plan["capability_gaps"],
            "suggestions": action_plan["suggestions"],
            "auto_evolution": {
                "requested": bool(auto_evolve and evolution_limit > 0),
                "triggered": False,
                "request_count": 0,
                "requests": [],
                "skipped": [],
            },
        }

        if auto_evolve and evolution_limit > 0:
            reflection["auto_evolution"] = await self._trigger_self_reflective_evolution(
                reflection["suggestions"],
                max_evolution_requests=evolution_limit,
            )

        self.state.performance_metrics = {
            "reflection_id": reflection_id,
            "created_at": created_at,
            **reflection["metrics"],
        }
        self._holon_service.record_self_reflection(self.holon_id, reflection)

        await self._safe_remember(
            content=summary,
            tags=["self_improvement", "analysis", "reflection"],
            importance=0.9,
        )
        await self._safe_persist_reflection_memory(
            reflection,
            phase_tag="analysis",
        )

        return reflection

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

    def _finalize_evolution_audit(
        self,
        request: EvolutionRequest,
        *,
        status: EvolutionStatus,
        phase: str,
        result_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Persist terminal request state so status survives timing races and restarts."""
        terminal_map = {
            EvolutionStatus.COMPLETED: "success",
            EvolutionStatus.FAILED: "failed",
            EvolutionStatus.REJECTED: "rejected",
        }
        audit_result = terminal_map.get(status)
        if audit_result is None:
            return

        current_state = self._holon_service.get_holon_state(self.holon_id)
        current_audit = current_state.get("evolution_audit")
        if isinstance(current_audit, dict):
            current_request_id = str(current_audit.get("request_id") or "").strip()
            if current_request_id and current_request_id != request.request_id:
                return

        payload = dict(result_payload or {})
        self._holon_service.update_evolution_audit(
            self.holon_id,
            patch={
                "request_id": request.request_id,
                "lifecycle": "completed",
                "phase": str(phase or "complete").strip() or "complete",
                "result": audit_result,
                "error": "" if status == EvolutionStatus.COMPLETED else str(error_message or "").strip(),
                "completed_at": str(request.completed_at or utc_now_iso()).strip(),
                "skill_id": str(payload.get("skill_id") or "").strip(),
                "attestation_id": str(payload.get("attestation_id") or "").strip(),
                "code_path": str(payload.get("code_path") or "").strip(),
                "test_path": str(payload.get("test_path") or "").strip(),
                "manifest_path": str(payload.get("manifest_path") or "").strip(),
                "llm": {"inflight": False},
            },
        )

    def _hydrate_evolution_request_from_audit(self, request_id: str) -> Optional[EvolutionRequest]:
        """Reconstruct request state from persisted audit metadata when memory is stale."""
        state_payload = self._holon_service.get_holon_state(self.holon_id)
        audit = state_payload.get("evolution_audit")
        if not isinstance(audit, dict):
            return None
        if str(audit.get("request_id") or "").strip() != str(request_id or "").strip():
            return None

        lifecycle = str(audit.get("lifecycle") or "").strip().lower()
        result = str(audit.get("result") or "").strip().lower()
        phase = str(audit.get("phase") or "").strip().lower()
        reason = str(state_payload.get("reason") or "").strip().lower()

        reconciled_status: Optional[EvolutionStatus]
        if lifecycle == "completed":
            if result in {"failed", "failure"} or reason == "evolution_failed_early":
                reconciled_status = EvolutionStatus.FAILED
            elif result == "rejected":
                reconciled_status = EvolutionStatus.REJECTED
            elif result in {"success", "completed"} or reason == "evolution_complete":
                reconciled_status = EvolutionStatus.COMPLETED
            else:
                reconciled_status = None
        elif phase == "red":
            reconciled_status = EvolutionStatus.RED_PHASE
        elif phase == "green":
            reconciled_status = EvolutionStatus.GREEN_PHASE
        elif phase == "verify":
            reconciled_status = EvolutionStatus.VERIFY_PHASE
        elif lifecycle == "pending":
            reconciled_status = EvolutionStatus.PENDING
        elif lifecycle == "running" or result == "in_progress":
            reconciled_status = EvolutionStatus.EVOLVING
        else:
            reconciled_status = None

        if reconciled_status is None:
            return None

        request = self._evolution_requests.get(request_id)
        if request is None:
            request = EvolutionRequest(
                request_id=request_id,
                holon_id=self.holon_id,
                skill_name=str(audit.get("skill_name") or "").strip(),
                description="",
                requirements=[],
                test_cases=[],
                parent_skills=[],
                status=reconciled_status,
                created_at=(
                    str(audit.get("started_at") or "").strip()
                    or str(state_payload.get("pending_at") or "").strip()
                    or utc_now_iso()
                ),
            )
            self._evolution_requests[request_id] = request
            if request_id not in self.state.evolution_requests:
                self.state.evolution_requests.append(request_id)
        else:
            request.status = reconciled_status

        completed_at = str(audit.get("completed_at") or state_payload.get("resumed_at") or "").strip()
        if completed_at:
            request.completed_at = completed_at

        error_message = str(audit.get("error") or "").strip()
        if error_message:
            request.error_message = error_message

        payload: Dict[str, Any] = {}
        for key in ("skill_id", "attestation_id", "code_path", "test_path", "manifest_path"):
            value = str(audit.get(key) or "").strip()
            if value:
                payload[key] = value
        if payload:
            request.result = payload

        return request

    def get_evolution_status(self, request_id: str) -> Optional[EvolutionRequest]:
        """Get status of an evolution request."""
        request = self._evolution_requests.get(request_id)
        hydrated = self._hydrate_evolution_request_from_audit(request_id)
        return hydrated or request

    def get_self_reflection(self, history_limit: int = 10) -> Dict[str, Any]:
        """Return the latest persisted self-reflection snapshot with bounded history."""
        limit = max(0, int(history_limit))
        state_payload = self._holon_service.get_holon_state(self.holon_id)
        reflection = state_payload.get("self_reflection")
        if not isinstance(reflection, dict):
            return {
                "status": "no_reflection",
                "history": [],
                "history_count": 0,
            }

        payload = dict(reflection)
        history = payload.get("history")
        history_items = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
        payload["history_count"] = len(history_items)
        payload["history"] = history_items[:limit] if limit > 0 else []
        return payload

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
        grace_window_seconds = max(1.0, min(5.0, max(1.0, poll_interval_seconds * 5)))
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                grace_deadline = time.monotonic() + grace_window_seconds
                while True:
                    final_status = self.get_evolution_status(request_id)
                    if final_status is not None and final_status.status in terminal:
                        return final_status
                    now = time.monotonic()
                    if now >= grace_deadline:
                        break
                    await asyncio.sleep(min(0.05, grace_deadline - now))
                raise TimeoutError(
                    f"Timed out waiting for evolution request {request_id} "
                    f"after {timeout_seconds:.1f}s"
                )
            await asyncio.sleep(min(poll_interval_seconds, remaining))

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
        self._assert_runtime_available(action="execute")
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
