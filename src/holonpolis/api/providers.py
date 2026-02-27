"""Provider management API routes for HolonPolis.

Provides endpoints for managing LLM providers:
- List providers (with key masking)
- Get provider details
- Create/update/delete providers
- Health checks
- List models
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from holonpolis.kernel.llm.provider_config import (
    get_provider_manager,
    ProviderConfig,
    MASKED_SECRET,
)

router = APIRouter(prefix="/providers", tags=["providers"])


class ProviderCreateRequest(BaseModel):
    """Request to create a new provider."""

    provider_id: str = Field(..., description="Unique provider identifier")
    provider_type: str = Field(..., description="Provider type (openai_compat, ollama, etc)")
    name: str = Field(..., description="Display name")
    base_url: str = Field("", description="API base URL")
    api_key: str = Field("", description="API key (will be encrypted at rest)")
    api_path: str = Field("/v1/chat/completions", description="API endpoint path")
    models_path: str = Field("/v1/models", description="Models endpoint path")
    timeout: int = Field(60, description="Request timeout in seconds")
    retries: int = Field(3, description="Number of retries")
    temperature: float = Field(0.7, description="Default temperature")
    max_tokens: int = Field(8192, description="Default max tokens")


class ProviderUpdateRequest(BaseModel):
    """Request to update a provider."""

    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None
    retries: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ProviderResponse(BaseModel):
    """Provider response (with masked secrets)."""

    provider_id: str
    provider_type: str
    name: str
    base_url: str
    api_key: str  # Masked
    api_path: str
    models_path: str
    timeout: int
    retries: int
    temperature: float
    max_tokens: int


class HealthCheckResponse(BaseModel):
    """Health check response."""

    healthy: bool
    status: Optional[int] = None
    message: str = ""
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    """Models list response."""

    success: bool
    models: List[str] = Field(default_factory=list)
    count: int = 0
    error: Optional[str] = None


@router.get("", response_model=List[ProviderResponse])
async def list_providers() -> List[Dict[str, Any]]:
    """List all configured providers (with API keys masked)."""
    manager = get_provider_manager()
    return manager.list_providers(mask_secrets=True)


@router.get("/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: str) -> Dict[str, Any]:
    """Get a specific provider configuration."""
    manager = get_provider_manager()
    provider = manager.get_provider(provider_id)

    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")

    return provider.to_dict(mask_secrets=True)


@router.post("", response_model=ProviderResponse)
async def create_provider(request: ProviderCreateRequest) -> Dict[str, Any]:
    """Create a new provider configuration."""
    manager = get_provider_manager()

    config = ProviderConfig(
        provider_id=request.provider_id,
        provider_type=request.provider_type,
        name=request.name,
        base_url=request.base_url,
        api_key=request.api_key,
        api_path=request.api_path,
        models_path=request.models_path,
        timeout=request.timeout,
        retries=request.retries,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    success, message = manager.add_provider(config)
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return config.to_dict(mask_secrets=True)


@router.patch("/{provider_id}", response_model=ProviderResponse)
async def update_provider(
    provider_id: str,
    request: ProviderUpdateRequest,
) -> Dict[str, Any]:
    """Update an existing provider configuration."""
    manager = get_provider_manager()

    # Build updates dict from non-None fields
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    success, message = manager.update_provider(provider_id, updates)
    if not success:
        raise HTTPException(status_code=404 if "not found" in message else 400, detail=message)

    provider = manager.get_provider(provider_id)
    return provider.to_dict(mask_secrets=True)


@router.delete("/{provider_id}")
async def delete_provider(provider_id: str) -> Dict[str, str]:
    """Delete a provider configuration."""
    manager = get_provider_manager()

    success, message = manager.delete_provider(provider_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)

    return {"message": message}


@router.post("/{provider_id}/health", response_model=HealthCheckResponse)
async def provider_health(provider_id: str) -> Dict[str, Any]:
    """Perform health check on a provider."""
    manager = get_provider_manager()
    return manager.health_check(provider_id)


@router.get("/{provider_id}/models", response_model=ModelsResponse)
async def list_models(provider_id: str) -> Dict[str, Any]:
    """List available models from a provider."""
    manager = get_provider_manager()
    return manager.list_models(provider_id)


@router.post("/{provider_id}/validate")
async def validate_provider_config(provider_id: str) -> Dict[str, Any]:
    """Validate a provider's configuration."""
    manager = get_provider_manager()
    provider = manager.get_provider(provider_id)

    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")

    valid, errors = manager._validate_config(provider)
    return {
        "provider_id": provider_id,
        "valid": valid,
        "errors": errors,
    }
