"""Holon Runtime - generic execution engine for any Holon.

A HolonRuntime is instantiated from a Blueprint and runs the conversation loop.
There are no specific Agent classes - all Holons are runtime instances.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.config import settings
from holonpolis.domain import Blueprint
from holonpolis.domain.memory import MemoryKind
from holonpolis.kernel.llm.llm_runtime import LLMConfig, LLMMessage, get_llm_runtime
from holonpolis.kernel.tools import (
    ToolExecutor,
    ToolChainStep,
    ToolChainResult,
    execute_tool,
)
from holonpolis.services.holon_service import HolonService
from holonpolis.services.memory_service import MemoryService

logger = structlog.get_logger()


@dataclass
class HolonState:
    """Current state of a Holon conversation."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    episode_count: int = 0
    total_tokens: int = 0


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
        skill_description: str,
        test_cases: List[Dict[str, Any]],
    ) -> str:
        """Request evolution of a new skill.

        This submits an evolution request to the EvolutionService.
        The actual evolution happens asynchronously.

        Args:
            skill_name: Name for the new skill
            skill_description: What the skill should do
            test_cases: Test cases the skill must pass

        Returns:
            evolution_id: The ID of the evolution request
        """
        # This would typically submit to an EvolutionService
        # For now, just record the request
        logger.info(
            "evolution_requested",
            holon_id=self.holon_id,
            skill_name=skill_name,
        )

        # Placeholder - full implementation would integrate with EvolutionService
        evolution_id = f"evo_{uuid.uuid4().hex[:12]}"

        return evolution_id

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

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            "holon_id": self.holon_id,
            "name": self.blueprint.name,
            "species": self.blueprint.species_id,
            "episode_count": self.state.episode_count,
            "total_tokens": self.state.total_tokens,
            "available_tools": len(self.get_available_tools()),
        }
