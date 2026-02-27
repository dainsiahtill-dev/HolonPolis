"""LLM Runtime - adapter layer for different LLM providers.

Based on HarborPilot patterns but simplified for HolonPolis needs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncGenerator
import time
import os

import structlog

logger = structlog.get_logger()


@dataclass
class LLMMessage:
    """A message in the conversation."""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[Dict]] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool messages


@dataclass
class LLMUsage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Response from LLM invocation."""
    content: str
    tool_calls: Optional[List[Dict]] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: int = 0
    model: str = ""
    thinking: Optional[str] = None  # For reasoning models
    raw: Optional[Dict] = None


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Invoke LLM with messages."""
        pass

    @abstractmethod
    async def invoke_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible provider."""

    def __init__(self):
        self._client = None

    def _get_client(self, config: Optional[LLMConfig] = None):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise RuntimeError("openai package not installed")

            cfg = config or LLMConfig()
            api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
            base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL")

            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        return self._client

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict]:
        """Convert internal messages to OpenAI format."""
        result = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.name:
                m["name"] = msg.name
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            result.append(m)
        return result

    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Invoke OpenAI API."""
        cfg = config or LLMConfig()
        client = self._get_client(cfg)

        start = time.time()

        try:
            params = {
                "model": cfg.model,
                "messages": self._convert_messages(messages),
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "top_p": cfg.top_p,
                "frequency_penalty": cfg.frequency_penalty,
                "presence_penalty": cfg.presence_penalty,
            }

            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            response = await client.chat.completions.create(**params)

            latency_ms = int((time.time() - start) * 1000)
            choice = response.choices[0]
            message = choice.message

            # Extract tool calls if present
            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            usage = LLMUsage()
            if response.usage:
                usage = LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                usage=usage,
                latency_ms=latency_ms,
                model=response.model,
                raw=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            logger.error("openai_invoke_failed", error=str(e))
            raise

    async def invoke_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI response."""
        cfg = config or LLMConfig()
        client = self._get_client(cfg)

        try:
            params = {
                "model": cfg.model,
                "messages": self._convert_messages(messages),
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": True,
            }

            if tools:
                params["tools"] = tools

            stream = await client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("openai_stream_failed", error=str(e))
            raise


class LLMRuntime:
    """Central runtime for LLM operations."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or OpenAIProvider()

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
        context_messages: Optional[List[LLMMessage]] = None,
    ) -> LLMResponse:
        """Simple chat interface."""
        messages = [LLMMessage(role="system", content=system_prompt)]

        if context_messages:
            messages.extend(context_messages)

        messages.append(LLMMessage(role="user", content=user_prompt))

        return await self.provider.invoke(messages, config, tools)

    async def chat_with_history(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Chat with full message history."""
        return await self.provider.invoke(messages, config, tools)

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response."""
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        async for chunk in self.provider.invoke_stream(messages, config):
            yield chunk


# Global runtime instance
_runtime: Optional[LLMRuntime] = None


def get_llm_runtime() -> LLMRuntime:
    """Get the global LLM runtime."""
    global _runtime
    if _runtime is None:
        _runtime = LLMRuntime()
    return _runtime
