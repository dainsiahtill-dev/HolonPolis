"""Holons router - management endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from holonpolis.api.dependencies import get_genesis_service
from holonpolis.domain import Blueprint
from holonpolis.services.genesis_service import GenesisService
from holonpolis.services.holon_service import HolonService

router = APIRouter(prefix="/holons", tags=["holons"])


class HolonSummary(BaseModel):
    """Summary of a Holon."""
    holon_id: str
    name: str
    species_id: str
    purpose: str
    status: str = "active"


class HolonDetail(BaseModel):
    """Detailed Holon info."""
    holon_id: str
    blueprint: dict
    stats: dict


class CreateHolonRequest(BaseModel):
    """Request to create a Holon manually."""
    name: str
    species_id: str = "generalist"
    purpose: str
    allowed_tools: List[str] = []


@router.get("", response_model=List[HolonSummary])
async def list_holons(
    genesis: GenesisService = Depends(get_genesis_service),
) -> List[HolonSummary]:
    """List all Holons."""
    holons = await genesis.list_all_holons()
    return [
        HolonSummary(
            holon_id=h["holon_id"],
            name=h["name"],
            species_id=h["species_id"],
            purpose=h["purpose"],
            status="active",
        )
        for h in holons
    ]


@router.get("/{holon_id}", response_model=HolonDetail)
async def get_holon(
    holon_id: str,
) -> HolonDetail:
    """Get detailed info about a Holon."""
    service = HolonService()

    try:
        blueprint = service.get_blueprint(holon_id)
        return HolonDetail(
            holon_id=holon_id,
            blueprint=blueprint.to_dict(),
            stats={
                "is_frozen": service.is_frozen(holon_id),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{holon_id}/freeze")
async def freeze_holon(
    holon_id: str,
) -> dict:
    """Freeze a Holon (prevent new episodes)."""
    service = HolonService()

    if not service.holon_exists(holon_id):
        raise HTTPException(status_code=404, detail="Holon not found")

    service.freeze_holon(holon_id)
    return {"status": "frozen", "holon_id": holon_id}


@router.post("/{holon_id}/resume")
async def resume_holon(
    holon_id: str,
) -> dict:
    """Resume a frozen Holon."""
    service = HolonService()

    if not service.holon_exists(holon_id):
        raise HTTPException(status_code=404, detail="Holon not found")

    service.resume_holon(holon_id)
    return {"status": "resumed", "holon_id": holon_id}


@router.delete("/{holon_id}")
async def delete_holon(
    holon_id: str,
) -> dict:
    """Delete a Holon permanently."""
    service = HolonService()

    if not service.holon_exists(holon_id):
        raise HTTPException(status_code=404, detail="Holon not found")

    service.delete_holon(holon_id)
    return {"status": "deleted", "holon_id": holon_id}


@router.get("/{holon_id}/memories")
async def get_holon_memories(
    holon_id: str,
    query: Optional[str] = None,
    top_k: int = 5,
) -> List[dict]:
    """Query memories of a Holon."""
    from holonpolis.services.memory_service import MemoryService

    service = MemoryService(holon_id)

    if query:
        memories = await service.recall(query, top_k=top_k)
    else:
        memories = []

    return memories


class LearnRepositoryRequest(BaseModel):
    """Request to learn from a repository."""
    repo_url: str
    branch: str = "main"
    depth: int = 3
    focus_areas: List[str] = []


class LearnRepositoryResponse(BaseModel):
    """Response from learning a repository."""
    success: bool
    holon_id: str
    repo_url: str
    repo_name: str
    languages: dict
    total_files: int
    total_lines: int
    key_patterns: List[str]
    architecture: str
    learnings: List[str]
    memories_created: int
    message: str


@router.post("/{holon_id}/learn-repository", response_model=LearnRepositoryResponse)
async def learn_repository(
    holon_id: str,
    request: LearnRepositoryRequest,
) -> LearnRepositoryResponse:
    """让 Holon 学习指定的代码仓库。

    示例:
    ```json
    {
        "repo_url": "https://github.com/vuejs/core",
        "branch": "main",
        "depth": 3,
        "focus_areas": ["architecture", "patterns", "testing"]
    }
    ```
    """
    from holonpolis.services.repository_learner import RepositoryLearningService

    service = RepositoryLearningService()

    # 检查 Holon 是否存在
    holon_svc = HolonService()
    if not holon_svc.holon_exists(holon_id):
        raise HTTPException(status_code=404, detail="Holon not found")

    try:
        result = await service.learn(
            holon_id=holon_id,
            repo_url=request.repo_url,
            branch=request.branch,
            depth=request.depth,
            focus_areas=request.focus_areas or None,
        )

        if result.success and result.analysis:
            return LearnRepositoryResponse(
                success=True,
                holon_id=holon_id,
                repo_url=request.repo_url,
                repo_name=result.analysis.repo_name,
                languages=result.analysis.languages,
                total_files=result.analysis.total_files,
                total_lines=result.analysis.total_lines,
                key_patterns=result.analysis.key_patterns,
                architecture=result.analysis.architecture,
                learnings=result.analysis.learnings,
                memories_created=result.memories_created,
                message=f"Successfully learned {result.analysis.repo_name}",
            )
        else:
            return LearnRepositoryResponse(
                success=False,
                holon_id=holon_id,
                repo_url=request.repo_url,
                repo_name="",
                languages={},
                total_files=0,
                total_lines=0,
                key_patterns=[],
                architecture="",
                learnings=[],
                memories_created=0,
                message=result.error_message or "Learning failed",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
