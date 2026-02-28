"""Holons router - management and capability endpoints."""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from holonpolis.api.dependencies import get_genesis_service, get_holon_manager_dep
from holonpolis.runtime.holon_manager import HolonManager
from holonpolis.runtime.holon_runtime import (
    CapabilityDeniedError,
    SkillPayloadValidationError,
)
from holonpolis.services.genesis_service import GenesisService
from holonpolis.services.holon_service import HolonService

router = APIRouter(prefix="/holons", tags=["holons"])

ERROR_400_BAD_REQUEST = {
    "description": "Request is syntactically valid but semantically invalid for this operation.",
    "content": {
        "application/json": {
            "example": {
                "detail": "Invalid request for current Holon state"
            }
        }
    },
}
ERROR_403_FORBIDDEN = {
    "description": "Operation denied by Holon boundary policy.",
    "content": {
        "application/json": {
            "example": {
                "detail": (
                    "Capability denied by boundary policy. "
                    "Requested any of: ['social.selection.execute', 'selection', 'execute']"
                )
            }
        }
    },
}
ERROR_404_HOLON = {
    "description": "Holon or requested resource was not found.",
    "content": {
        "application/json": {
            "example": {
                "detail": "Holon not found"
            }
        }
    },
}
ERROR_422_PAYLOAD = {
    "description": "Input payload failed schema validation.",
    "content": {
        "application/json": {
            "example": {
                "detail": "payload missing required fields: a, b"
            }
        }
    },
}


def _to_json_safe(value: Any) -> Any:
    """Convert nested values into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    if is_dataclass(value):
        return _to_json_safe(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_json_safe(to_dict())
    return str(value)


def _require_holon(holon_id: str) -> HolonService:
    service = HolonService()
    if not service.holon_exists(holon_id):
        raise HTTPException(status_code=404, detail="Holon not found")
    return service


def _get_runtime(holon_id: str, manager: HolonManager):
    _require_holon(holon_id)
    return manager.get_runtime(holon_id)


def _enforce_capability(
    runtime,
    capability: str,
    aliases: Optional[List[str]] = None,
) -> None:
    try:
        runtime.enforce_capability(capability, aliases=aliases)
    except CapabilityDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


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
    allowed_tools: List[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Specialized Analyst",
                "species_id": "specialist",
                "purpose": "Analyze architecture quality for backend systems",
                "allowed_tools": ["search", "analyze", "social.competition.execute"],
            }
        }
    }


class EvolutionRequestBody(BaseModel):
    """Request body for triggering skill evolution."""

    skill_name: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    parent_skills: List[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "skill_name": "DataNormalizer",
                "description": "Normalize and validate incoming customer data",
                "requirements": ["accept dict input", "sanitize fields", "return normalized dict"],
                "test_cases": [
                    {
                        "description": "basic normalize",
                        "function": "execute",
                        "input": {"payload": {"name": " ALICE "}},
                        "expected": {"name": "alice"},
                    }
                ],
                "parent_skills": [],
            }
        }
    }


class SkillExecuteRequest(BaseModel):
    """Request body for executing a persisted evolved skill."""

    payload: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "payload": {
                    "a": 2,
                    "b": 3,
                }
            }
        }
    }


class OfferSkillRequest(BaseModel):
    """Request body for publishing a skill offer to marketplace."""

    skill_name: str
    description: str
    price_per_use: float = 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "skill_name": "Code Review",
                "description": "Review pull requests for security and performance",
                "price_per_use": 80.0,
            }
        }
    }


class SelectionRequest(BaseModel):
    """Request body for market selection run."""

    threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "threshold": 0.7,
            }
        }
    }


class RelationshipRequest(BaseModel):
    """Request body for creating a social relationship."""

    target_holon_id: str
    relationship_type: str = "peer"
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_holon_id": "holon_beta",
                "relationship_type": "mentor",
                "strength": 0.75,
                "trust_score": 0.8,
                "metadata": {"context": "pair-programming on migration"},
            }
        }
    }


class CollaborationRequest(BaseModel):
    """Request body for collaboration initiation."""

    task_name: str
    task_description: str
    collaborator_ids: List[str] = Field(default_factory=list)
    subtasks: List[Dict[str, str]] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_name": "Build Landing Page",
                "task_description": "Create and test a responsive landing page",
                "collaborator_ids": ["holon_design", "holon_test"],
                "subtasks": [
                    {"name": "Design", "description": "Create UI design"},
                    {"name": "Implement", "description": "Build page components"},
                    {"name": "Test", "description": "Run QA checks"},
                ],
            }
        }
    }


class CompetitionRequest(BaseModel):
    """Request body for competition participation."""

    task_description: str
    competitors: List[str] = Field(default_factory=list)
    evaluation_criteria: Dict[str, float] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_description": "Generate robust login form component",
                "competitors": ["holon_a", "holon_b", "holon_c"],
                "evaluation_criteria": {"accuracy": 0.4, "speed": 0.3, "quality": 0.3},
            }
        }
    }


class LearnRepositoryRequest(BaseModel):
    """Request to learn from a repository."""

    repo_url: str
    branch: str = "main"
    depth: int = 3
    focus_areas: List[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "repo_url": "https://github.com/fastapi/fastapi",
                "branch": "master",
                "depth": 3,
                "focus_areas": ["architecture", "patterns", "testing"],
            }
        }
    }


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
async def get_holon(holon_id: str) -> HolonDetail:
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
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{holon_id}/freeze")
async def freeze_holon(holon_id: str) -> dict:
    """Freeze a Holon (prevent new episodes)."""
    service = _require_holon(holon_id)
    service.freeze_holon(holon_id)
    return {"status": "frozen", "holon_id": holon_id}


@router.post("/{holon_id}/resume")
async def resume_holon(holon_id: str) -> dict:
    """Resume a frozen Holon."""
    service = _require_holon(holon_id)
    service.resume_holon(holon_id)
    return {"status": "resumed", "holon_id": holon_id}


@router.delete("/{holon_id}")
async def delete_holon(
    holon_id: str,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> dict:
    """Delete a Holon permanently."""
    service = _require_holon(holon_id)
    manager.release_runtime(holon_id)
    service.delete_holon(holon_id)
    return {"status": "deleted", "holon_id": holon_id}


@router.get("/{holon_id}/memories")
async def get_holon_memories(
    holon_id: str,
    query: Optional[str] = None,
    top_k: int = Query(default=5, ge=1, le=50),
) -> List[dict]:
    """Query memories of a Holon."""
    from holonpolis.services.memory_service import MemoryService

    _require_holon(holon_id)
    service = MemoryService(holon_id)
    if query:
        memories = await service.recall(query, top_k=top_k)
    else:
        memories = []
    return memories


@router.post(
    "/{holon_id}/evolution/requests",
    responses={
        403: ERROR_403_FORBIDDEN,
        404: ERROR_404_HOLON,
        422: ERROR_422_PAYLOAD,
    },
)
async def request_evolution(
    holon_id: str,
    request: EvolutionRequestBody,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Request a new skill evolution through RGV."""
    runtime = _get_runtime(holon_id, manager)
    _enforce_capability(runtime, "evolution.request", aliases=["evolve", "execute"])
    evo = await runtime.request_evolution(
        skill_name=request.skill_name,
        description=request.description,
        requirements=request.requirements,
        test_cases=request.test_cases,
        parent_skills=request.parent_skills,
    )
    return {
        "request_id": evo.request_id,
        "holon_id": evo.holon_id,
        "skill_name": evo.skill_name,
        "status": evo.status.value,
        "created_at": evo.created_at,
    }


@router.get(
    "/{holon_id}/evolution/requests/{request_id}",
    responses={404: ERROR_404_HOLON},
)
async def get_evolution_status(
    holon_id: str,
    request_id: str,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Get status of an evolution request."""
    runtime = _get_runtime(holon_id, manager)
    status = runtime.get_evolution_status(request_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Evolution request not found: {request_id}")
    return {
        "request_id": status.request_id,
        "holon_id": status.holon_id,
        "skill_name": status.skill_name,
        "description": status.description,
        "requirements": status.requirements,
        "status": status.status.value,
        "created_at": status.created_at,
        "completed_at": status.completed_at,
        "result": _to_json_safe(status.result),
        "error_message": status.error_message,
    }


@router.get(
    "/{holon_id}/skills",
    responses={404: ERROR_404_HOLON},
)
async def list_skills(
    holon_id: str,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> List[Dict[str, Any]]:
    """List persisted evolved skills for a Holon."""
    runtime = _get_runtime(holon_id, manager)
    return runtime.list_skills()


@router.post(
    "/{holon_id}/skills/{skill_name_or_id}/execute",
    responses={
        400: ERROR_400_BAD_REQUEST,
        403: ERROR_403_FORBIDDEN,
        404: ERROR_404_HOLON,
        422: ERROR_422_PAYLOAD,
    },
)
async def execute_skill(
    holon_id: str,
    skill_name_or_id: str,
    request: SkillExecuteRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Execute a persisted evolved skill."""
    runtime = _get_runtime(holon_id, manager)
    _enforce_capability(runtime, "skill.execute", aliases=["execute"])
    try:
        result = await runtime.execute_skill(
            skill_name_or_id=skill_name_or_id,
            payload=request.payload,
        )
    except CapabilityDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except SkillPayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "holon_id": holon_id,
        "skill": skill_name_or_id,
        "result": _to_json_safe(result),
    }


@router.post(
    "/{holon_id}/market/offers",
    responses={404: ERROR_404_HOLON},
)
async def offer_skill(
    holon_id: str,
    request: OfferSkillRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """List a skill offer in marketplace."""
    runtime = _get_runtime(holon_id, manager)
    offer_id = await runtime.offer_skill(
        skill_name=request.skill_name,
        description=request.description,
        price_per_use=request.price_per_use,
    )
    return {
        "holon_id": holon_id,
        "offer_id": offer_id,
        "skill_name": request.skill_name,
        "price_per_use": request.price_per_use,
    }


@router.get(
    "/{holon_id}/market/providers",
    responses={404: ERROR_404_HOLON},
)
async def find_skill_providers(
    holon_id: str,
    skill_query: str,
    max_price: Optional[float] = None,
    top_k: int = Query(default=3, ge=1, le=20),
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> List[Dict[str, Any]]:
    """Find providers from marketplace for required skill."""
    runtime = _get_runtime(holon_id, manager)
    return await runtime.find_skill_providers(
        skill_query=skill_query,
        max_price=max_price,
        top_k=top_k,
    )


@router.get(
    "/{holon_id}/market/stats",
    responses={404: ERROR_404_HOLON},
)
async def get_market_stats(
    holon_id: str,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Get aggregated market statistics."""
    runtime = _get_runtime(holon_id, manager)
    return runtime.get_market_stats()


@router.post(
    "/{holon_id}/selection",
    responses={
        403: ERROR_403_FORBIDDEN,
        404: ERROR_404_HOLON,
        422: ERROR_422_PAYLOAD,
    },
)
async def run_selection(
    holon_id: str,
    request: SelectionRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Run market selection process."""
    runtime = _get_runtime(holon_id, manager)
    _enforce_capability(
        runtime,
        "social.selection.execute",
        aliases=["selection", "execute"],
    )
    result = await runtime.run_selection(threshold=request.threshold)
    return _to_json_safe(result)


@router.get(
    "/{holon_id}/collaborators",
    responses={404: ERROR_404_HOLON},
)
async def find_collaborators(
    holon_id: str,
    skill_needed: str,
    min_reputation: float = Query(default=0.3, ge=0.0, le=1.0),
    top_k: int = Query(default=5, ge=1, le=20),
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> List[Dict[str, Any]]:
    """Find potential collaborators for skill need."""
    runtime = _get_runtime(holon_id, manager)
    return await runtime.find_collaborators(
        skill_needed=skill_needed,
        min_reputation=min_reputation,
        top_k=top_k,
    )


@router.post(
    "/{holon_id}/collaborations",
    responses={404: ERROR_404_HOLON, 422: ERROR_422_PAYLOAD},
)
async def collaborate(
    holon_id: str,
    request: CollaborationRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Start a collaboration task with other Holons."""
    runtime = _get_runtime(holon_id, manager)
    return await runtime.collaborate(
        task_name=request.task_name,
        task_description=request.task_description,
        collaborator_ids=request.collaborator_ids,
        subtasks=request.subtasks,
    )


@router.post(
    "/{holon_id}/competitions",
    responses={
        403: ERROR_403_FORBIDDEN,
        404: ERROR_404_HOLON,
        422: ERROR_422_PAYLOAD,
    },
)
async def compete(
    holon_id: str,
    request: CompetitionRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Participate in a competition task."""
    runtime = _get_runtime(holon_id, manager)
    _enforce_capability(
        runtime,
        "social.competition.execute",
        aliases=["competition", "execute"],
    )
    return await runtime.compete(
        task_description=request.task_description,
        competitors=request.competitors,
        evaluation_criteria=request.evaluation_criteria or None,
    )


@router.post(
    "/{holon_id}/relationships",
    responses={
        400: ERROR_400_BAD_REQUEST,
        404: ERROR_404_HOLON,
        422: ERROR_422_PAYLOAD,
    },
)
async def register_relationship(
    holon_id: str,
    request: RelationshipRequest,
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Create a social relationship edge."""
    runtime = _get_runtime(holon_id, manager)
    try:
        relationship_id = await runtime.register_relationship(
            target_holon_id=request.target_holon_id,
            relationship_type=request.relationship_type,
            strength=request.strength,
            trust_score=request.trust_score,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "relationship_id": relationship_id,
        "source_holon_id": holon_id,
        "target_holon_id": request.target_holon_id,
        "relationship_type": request.relationship_type.lower(),
    }


@router.get(
    "/{holon_id}/trust/{target_holon_id}",
    responses={404: ERROR_404_HOLON},
)
async def propagate_trust(
    holon_id: str,
    target_holon_id: str,
    max_hops: int = Query(default=2, ge=1, le=5),
    manager: HolonManager = Depends(get_holon_manager_dep),
) -> Dict[str, Any]:
    """Propagate trust score from source to target holon."""
    runtime = _get_runtime(holon_id, manager)
    trust_score = await runtime.propagate_trust(
        target_holon_id=target_holon_id,
        max_hops=max_hops,
    )
    return {
        "source_holon_id": holon_id,
        "target_holon_id": target_holon_id,
        "max_hops": max_hops,
        "trust_score": trust_score,
    }


@router.post("/{holon_id}/learn-repository", response_model=LearnRepositoryResponse)
async def learn_repository(
    holon_id: str,
    request: LearnRepositoryRequest,
) -> LearnRepositoryResponse:
    """让 Holon 学习指定的代码仓库。"""
    from holonpolis.services.repository_learner import RepositoryLearningService

    service = RepositoryLearningService()

    # 检查 Holon 是否存在
    _require_holon(holon_id)

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

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
