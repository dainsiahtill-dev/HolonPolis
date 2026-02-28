"""Chat router - main user entry point.

/chat -> Genesis -> Route/Spawn -> HolonRuntime
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from holonpolis.api.dependencies import get_genesis_service, get_holon_manager_dep
from holonpolis.runtime.holon_manager import HolonManager
from holonpolis.services.genesis_service import GenesisService, RouteResult
from holonpolis.services.holon_service import HolonUnavailableError

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    conversation_id: Optional[str] = None
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    """Chat response."""
    content: str
    holon_id: str
    holon_name: Optional[str] = None
    episode_id: Optional[str] = None
    route_decision: str
    latency_ms: int = 0


class ConversationMessage(BaseModel):
    """A message in conversation history."""
    role: str
    content: str


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    genesis: GenesisService = Depends(get_genesis_service),
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> ChatResponse:
    """Main chat endpoint.

    Flow:
    1. Genesis analyzes request
    2. Routes to existing Holon or spawns new one
    3. HolonRuntime processes message
    4. Returns response
    """
    import time
    start = time.time()

    # Step 1 & 2: Genesis routing decision
    route_result = await genesis.route_or_spawn(
        user_request=request.message,
    )

    if route_result.decision == "deny":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Request denied",
                "reason": route_result.reasoning,
                "suggestion": route_result.message,
            }
        )

    if route_result.decision == "clarify":
        return ChatResponse(
            content=route_result.message or "Could you clarify your request?",
            holon_id="genesis",
            route_decision="clarify",
            latency_ms=int((time.time() - start) * 1000),
        )

    if route_result.holon_id is None:
        raise HTTPException(
            status_code=500,
            detail="Routing failed: no holon_id",
        )

    # Step 3: Process with HolonRuntime
    try:
        result = await manager.chat(
            holon_id=route_result.holon_id,
            user_message=request.message,
        )

        if route_result.route_id:
            await genesis.genesis_memory.update_route_outcome(route_result.route_id, "success")
        await genesis.genesis_memory.update_holon_stats(
            holon_id=route_result.holon_id,
            episode_increment=1,
            success=True,
        )

        latency_ms = int((time.time() - start) * 1000)

        return ChatResponse(
            content=result["content"],
            holon_id=route_result.holon_id,
            holon_name=route_result.blueprint.name if route_result.blueprint else None,
            episode_id=result.get("episode_id"),
            route_decision=route_result.decision,
            latency_ms=latency_ms,
        )

    except Exception as e:
        if route_result.route_id:
            await genesis.genesis_memory.update_route_outcome(route_result.route_id, "failure")
        await genesis.genesis_memory.update_holon_stats(
            holon_id=route_result.holon_id,
            episode_increment=1,
            success=False,
        )
        if isinstance(e, HolonUnavailableError):
            raise HTTPException(
                status_code=409,
                detail=str(e),
            )
        raise HTTPException(
            status_code=500,
            detail=f"Holon processing failed: {str(e)}",
        )


@router.post("/{holon_id}")
async def chat_with_specific_holon(
    holon_id: str,
    request: ChatRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> ChatResponse:
    """Chat directly with a specific Holon (bypasses Genesis routing)."""
    import time
    start = time.time()

    try:
        result = await manager.chat(
            holon_id=holon_id,
            user_message=request.message,
        )

        latency_ms = int((time.time() - start) * 1000)

        return ChatResponse(
            content=result["content"],
            holon_id=holon_id,
            episode_id=result.get("episode_id"),
            route_decision="direct",
            latency_ms=latency_ms,
        )

    except Exception as e:
        if isinstance(e, HolonUnavailableError):
            raise HTTPException(
                status_code=409,
                detail=str(e),
            )
        raise HTTPException(
            status_code=500,
            detail=f"Holon processing failed: {str(e)}",
        )
