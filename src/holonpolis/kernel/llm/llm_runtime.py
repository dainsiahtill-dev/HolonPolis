"""LLM Runtime - multi-provider adapter layer for HolonPolis."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import structlog

from holonpolis.config import settings

from .provider_config import load_provider_bundle

logger = structlog.get_logger()


def _join_url(base_url: str, path: str) -> str:
    raw_path = str(path or "").strip()
    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        return raw_path

    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return raw_path
    normalized_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
    return f"{base}{normalized_path}"


@dataclass
class LLMMessage:
    """A message in the conversation."""

    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


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
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: int = 0
    model: str = ""
    thinking: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    """Per-call configuration for LLM calls."""

    provider_id: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for all runtime providers."""

    @abstractmethod
    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Invoke model with the given messages."""

    async def invoke_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Default stream fallback: return full response as one chunk."""
        result = await self.invoke(messages, config, provider_config, tools)
        if result.content:
            yield result.content


class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI-compatible provider (OpenAI, Kimi-compatible endpoints, etc.)."""

    def _resolve_auth(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> Tuple[str, str]:
        api_key = (
            (config.api_key if config else None)
            or provider_config.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = (
            (config.base_url if config else None)
            or provider_config.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if not api_key:
            raise ValueError("missing_api_key_for_openai_compat_provider")
        return str(api_key), str(base_url)

    def _resolve_model(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> str:
        model = (config.model if config else None) or provider_config.get("model") or settings.openai_model
        return str(model)

    @staticmethod
    def _convert_messages(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for message in messages:
            item: Dict[str, Any] = {"role": message.role, "content": message.content}
            if message.name:
                item["name"] = message.name
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            payload.append(item)
        return payload

    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        from openai import AsyncOpenAI

        api_key, base_url = self._resolve_auth(config, provider_config)
        model = self._resolve_model(config, provider_config)
        timeout = (config.timeout_seconds if config else None) or provider_config.get("timeout")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        start = time.time()
        params: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": config.temperature if config else provider_config.get("temperature", 0.7),
            "max_tokens": config.max_tokens if config else provider_config.get("max_tokens", 4096),
            "top_p": config.top_p if config else provider_config.get("top_p", 1.0),
            "frequency_penalty": (
                config.frequency_penalty if config else provider_config.get("frequency_penalty", 0.0)
            ),
            "presence_penalty": (
                config.presence_penalty if config else provider_config.get("presence_penalty", 0.0)
            ),
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if config and config.extra:
            params.update(config.extra)

        response = await client.chat.completions.create(**params)
        latency_ms = int((time.time() - start) * 1000)

        choice = response.choices[0]
        message = choice.message
        usage = LLMUsage()
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        tool_calls = None
        if getattr(message, "tool_calls", None):
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

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=latency_ms,
            model=response.model or model,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def invoke_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI

        api_key, base_url = self._resolve_auth(config, provider_config)
        model = self._resolve_model(config, provider_config)
        timeout = (config.timeout_seconds if config else None) or provider_config.get("timeout")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        params: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": config.temperature if config else provider_config.get("temperature", 0.7),
            "max_tokens": config.max_tokens if config else provider_config.get("max_tokens", 4096),
            "stream": True,
        }
        if tools:
            params["tools"] = tools
        if config and config.extra:
            params.update(config.extra)

        stream = await client.chat.completions.create(**params)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(BaseLLMProvider):
    """Native Ollama provider via /api/chat."""

    def _resolve_base_url(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> str:
        return str(
            (config.base_url if config else None)
            or provider_config.get("base_url")
            or "http://127.0.0.1:11434"
        )

    def _resolve_model(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> str:
        return str((config.model if config else None) or provider_config.get("model") or "qwen2.5-coder:14b")

    @staticmethod
    def _convert_messages(messages: List[LLMMessage]) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for message in messages:
            role = message.role if message.role in {"system", "user", "assistant"} else "user"
            payload.append({"role": role, "content": message.content})
        return payload

    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        del tools
        import httpx

        base_url = self._resolve_base_url(config, provider_config)
        model = self._resolve_model(config, provider_config)
        api_path = str(provider_config.get("api_path") or "/api/chat")
        timeout = (config.timeout_seconds if config else None) or provider_config.get("timeout", 120)
        url = _join_url(base_url, api_path)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": False,
        }
        options = (config.extra if config else {}).get("options")
        if options is None:
            options = provider_config.get("options")
        if isinstance(options, dict):
            payload["options"] = options

        start = time.time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        output = ""
        if isinstance(data, dict):
            if isinstance(data.get("message"), dict):
                output = str(data["message"].get("content") or "")
            elif data.get("response") is not None:
                output = str(data.get("response") or "")

        usage = LLMUsage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0) if isinstance(data, dict) else 0,
            completion_tokens=int(data.get("eval_count") or 0) if isinstance(data, dict) else 0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return LLMResponse(
            content=output,
            usage=usage,
            latency_ms=int((time.time() - start) * 1000),
            model=model,
            raw=data if isinstance(data, dict) else None,
        )

    async def invoke_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        del tools
        import httpx

        base_url = self._resolve_base_url(config, provider_config)
        model = self._resolve_model(config, provider_config)
        api_path = str(provider_config.get("api_path") or "/api/chat")
        timeout = (config.timeout_seconds if config else None) or provider_config.get("timeout", 120)
        url = _join_url(base_url, api_path)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    raw = str(line or "").strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        if item.get("done") is True:
                            break
                        message = item.get("message")
                        if isinstance(message, dict) and message.get("content"):
                            yield str(message["content"])
                        elif item.get("response"):
                            yield str(item["response"])


class AnthropicCompatProvider(BaseLLMProvider):
    """Anthropic-compatible provider via /v1/messages."""

    def _resolve_auth(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> Tuple[str, str]:
        api_key = (
            (config.api_key if config else None)
            or provider_config.get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        base_url = (
            (config.base_url if config else None)
            or provider_config.get("base_url")
            or "https://api.anthropic.com/v1"
        )
        if not api_key:
            raise ValueError("missing_api_key_for_anthropic_provider")
        return str(api_key), str(base_url)

    def _resolve_model(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> str:
        return str(
            (config.model if config else None)
            or provider_config.get("model")
            or "claude-3-5-sonnet-latest"
        )

    @staticmethod
    def _convert_messages(messages: List[LLMMessage]) -> Tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            role = "assistant" if message.role == "assistant" else "user"
            anthropic_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": message.content}],
                }
            )

        system_prompt = "\n\n".join(system_parts).strip()
        return system_prompt, anthropic_messages

    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        del tools
        import httpx

        api_key, base_url = self._resolve_auth(config, provider_config)
        model = self._resolve_model(config, provider_config)
        api_path = str(provider_config.get("api_path") or "/v1/messages")
        version = str(provider_config.get("anthropic_version") or "2023-06-01")
        timeout = (config.timeout_seconds if config else None) or provider_config.get("timeout", 120)
        url = _join_url(base_url, api_path)

        system_prompt, anthropic_messages = self._convert_messages(messages)
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": config.max_tokens if config else provider_config.get("max_tokens", 1024),
            "temperature": config.temperature if config else provider_config.get("temperature", 0.2),
            "messages": anthropic_messages or [{"role": "user", "content": [{"type": "text", "text": ""}]}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if config and config.extra:
            payload.update(config.extra)

        headers = {
            "content-type": "application/json",
            "anthropic-version": version,
            "x-api-key": api_key,
        }

        start = time.time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        chunks: List[str] = []
        for block in data.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text") or ""))
        output = "".join(chunks)

        usage_data = data.get("usage") if isinstance(data, dict) else {}
        usage = LLMUsage(
            prompt_tokens=int(usage_data.get("input_tokens") or 0) if isinstance(usage_data, dict) else 0,
            completion_tokens=int(usage_data.get("output_tokens") or 0) if isinstance(usage_data, dict) else 0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return LLMResponse(
            content=output,
            usage=usage,
            latency_ms=int((time.time() - start) * 1000),
            model=model,
            raw=data if isinstance(data, dict) else None,
        )


class GeminiAPIProvider(BaseLLMProvider):
    """Gemini REST provider via generateContent endpoint."""

    def _resolve_auth(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> Tuple[str, str]:
        api_key = (
            (config.api_key if config else None)
            or provider_config.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        base_url = (
            (config.base_url if config else None)
            or provider_config.get("base_url")
            or "https://generativelanguage.googleapis.com"
        )
        if not api_key:
            raise ValueError("missing_api_key_for_gemini_provider")
        return str(api_key), str(base_url)

    def _resolve_model(self, config: Optional[LLMConfig], provider_config: Dict[str, Any]) -> str:
        return str((config.model if config else None) or provider_config.get("model") or "gemini-1.5-pro")

    @staticmethod
    def _messages_to_prompt(messages: List[LLMMessage]) -> str:
        lines = []
        for message in messages:
            lines.append(f"{message.role}: {message.content}")
        return "\n".join(lines)

    async def invoke(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig],
        provider_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        del tools
        import httpx

        api_key, base_url = self._resolve_auth(config, provider_config)
        model = self._resolve_model(config, provider_config)
        api_path_template = str(provider_config.get("api_path") or "/v1beta/models/{model}:generateContent")
        timeout = (config.timeout_seconds if config else None) or provider_config.get("timeout", 120)

        api_path = api_path_template.format(model=model)
        url = _join_url(base_url, api_path)
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}key={api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": self._messages_to_prompt(messages)}],
                }
            ],
            "generationConfig": {
                "temperature": config.temperature if config else provider_config.get("temperature", 0.7),
                "maxOutputTokens": config.max_tokens if config else provider_config.get("max_tokens", 4096),
                "topP": config.top_p if config else provider_config.get("top_p", 1.0),
            },
        }
        if config and config.extra:
            payload.update(config.extra)

        start = time.time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers={"content-type": "application/json"})
            response.raise_for_status()
            data = response.json()

        output_parts: List[str] = []
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list):
                        for part in parts:
                            if isinstance(part, dict) and part.get("text") is not None:
                                output_parts.append(str(part.get("text")))

        return LLMResponse(
            content="".join(output_parts),
            usage=LLMUsage(),
            latency_ms=int((time.time() - start) * 1000),
            model=model,
            raw=data if isinstance(data, dict) else None,
        )


class ProviderManager:
    """Registry for runtime provider implementations."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register_provider("openai_compat", OpenAICompatProvider())
        self.register_provider("ollama", OllamaProvider())
        self.register_provider("anthropic_compat", AnthropicCompatProvider())
        self.register_provider("gemini_api", GeminiAPIProvider())

    def register_provider(self, provider_type: str, provider: BaseLLMProvider) -> None:
        self._providers[str(provider_type)] = provider

    def get_provider(self, provider_type: str) -> Optional[BaseLLMProvider]:
        return self._providers.get(str(provider_type))

    def list_provider_types(self) -> List[str]:
        return sorted(self._providers.keys())


class LLMRuntime:
    """Central runtime for LLM operations with provider routing."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        provider_manager: Optional[ProviderManager] = None,
        provider_bundle: Optional[Dict[str, Any]] = None,
    ):
        self.provider_manager = provider_manager or ProviderManager()

        if provider is not None:
            self.provider_manager.register_provider("custom", provider)
            self.provider_bundle = {
                "default_provider_id": "custom",
                "providers": {"custom": {"type": "custom", "model": settings.openai_model}},
            }
        else:
            self.provider_bundle = provider_bundle or load_provider_bundle()

        self.default_provider_id = str(self.provider_bundle.get("default_provider_id") or "openai")
        providers = self.provider_bundle.get("providers")
        self.provider_configs: Dict[str, Dict[str, Any]] = providers if isinstance(providers, dict) else {}

    def reload_provider_bundle(self, provider_bundle: Optional[Dict[str, Any]] = None) -> None:
        """Reload provider config bundle from explicit data or configured sources."""
        self.provider_bundle = provider_bundle or load_provider_bundle()
        self.default_provider_id = str(self.provider_bundle.get("default_provider_id") or "openai")
        providers = self.provider_bundle.get("providers")
        self.provider_configs = providers if isinstance(providers, dict) else {}

    def get_provider_bundle(self) -> Dict[str, Any]:
        return {
            "default_provider_id": self.default_provider_id,
            "providers": dict(self.provider_configs),
        }

    def _resolve_provider(
        self, config: Optional[LLMConfig]
    ) -> Tuple[BaseLLMProvider, Dict[str, Any], str]:
        provider_id = (config.provider_id if config else None) or self.default_provider_id
        provider_cfg = self.provider_configs.get(str(provider_id))

        if provider_cfg is None and self.provider_manager.get_provider(str(provider_id)) is not None:
            provider_cfg = {"type": str(provider_id)}

        if provider_cfg is None:
            raise ValueError(f"unknown_provider_id: {provider_id}")

        provider_type = str(provider_cfg.get("type") or provider_id)
        provider = self.provider_manager.get_provider(provider_type)
        if provider is None:
            raise ValueError(f"unsupported_provider_type: {provider_type}")

        return provider, provider_cfg, str(provider_id)

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        context_messages: Optional[List[LLMMessage]] = None,
    ) -> LLMResponse:
        messages = [LLMMessage(role="system", content=system_prompt)]
        if context_messages:
            messages.extend(context_messages)
        messages.append(LLMMessage(role="user", content=user_prompt))
        return await self.chat_with_history(messages, config=config, tools=tools)

    async def chat_with_history(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        provider, provider_cfg, provider_id = self._resolve_provider(config)
        logger.debug("llm_provider_selected", provider_id=provider_id, provider_type=provider_cfg.get("type"))
        return await provider.invoke(messages, config, provider_cfg, tools)

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        provider, provider_cfg, _ = self._resolve_provider(config)
        async for chunk in provider.invoke_stream(messages, config, provider_cfg, tools):
            yield chunk


# Backward-compat alias
OpenAIProvider = OpenAICompatProvider


# Global runtime instance
_runtime: Optional[LLMRuntime] = None


def get_llm_runtime() -> LLMRuntime:
    """Get the global LLM runtime."""
    global _runtime
    if _runtime is None:
        _runtime = LLMRuntime()
    return _runtime


def reset_llm_runtime() -> None:
    """Reset global runtime (mainly for testing)."""
    global _runtime
    _runtime = None

