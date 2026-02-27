"""Tests for LLM runtime provider registry routing."""

import pytest

from holonpolis.kernel.llm.llm_runtime import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMRuntime,
    LLMUsage,
    ProviderManager,
)


class DummyProvider(BaseLLMProvider):
    async def invoke(self, messages, config, provider_config, tools=None):
        del tools
        return LLMResponse(
            content=f"{provider_config.get('name')}::{messages[-1].content}",
            usage=LLMUsage(total_tokens=1),
            model=str(config.model if config else provider_config.get("model") or ""),
        )


@pytest.mark.asyncio
async def test_runtime_routes_to_registered_provider():
    manager = ProviderManager()
    manager.register_provider("dummy", DummyProvider())

    runtime = LLMRuntime(
        provider_manager=manager,
        provider_bundle={
            "default_provider_id": "dummy_id",
            "providers": {
                "dummy_id": {
                    "type": "dummy",
                    "name": "dummy-provider",
                    "model": "dummy-model",
                }
            },
        },
    )

    result = await runtime.chat(
        system_prompt="sys",
        user_prompt="hello",
        config=LLMConfig(provider_id="dummy_id", model="dummy-model"),
    )

    assert "dummy-provider::hello" == result.content


@pytest.mark.asyncio
async def test_runtime_raises_for_unknown_provider_id():
    runtime = LLMRuntime(
        provider_bundle={
            "default_provider_id": "openai",
            "providers": {"openai": {"type": "openai_compat"}},
        }
    )
    with pytest.raises(ValueError, match="unknown_provider_id"):
        await runtime.chat(
            system_prompt="sys",
            user_prompt="hello",
            config=LLMConfig(provider_id="missing"),
        )

