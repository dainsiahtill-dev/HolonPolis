"""Collaboration Service - 管理 Holon 之间的协作。

核心功能:
1. 任务分解与分配
2. 协作协调
3. 结果汇总
4. 冲突解决
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import structlog

from holonpolis.config import settings
from holonpolis.domain.social import (
    CollaborationState,
    CollaborationTask,
    RelationshipType,
    Reputation,
    SocialRelationship,
    SocialGraph,
    SubTask,
)
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.runtime.holon_runtime import HolonRuntime
from holonpolis.services.holon_service import HolonService, HolonUnavailableError
from holonpolis.services.social_state_store import SocialStateStore

logger = structlog.get_logger()

# Process-wide state to avoid losing collaboration data across service instances.
_GLOBAL_ACTIVE_COLLABORATIONS: Dict[str, CollaborationTask] = {}
_GLOBAL_SOCIAL_GRAPH = SocialGraph()
_GLOBAL_COLLAB_STATE_ROOT: Optional[str] = None


class CollaborationService:
    """Holon 协作服务。

    协调多个 Holon 共同完成复杂任务。
    """

    def __init__(self):
        self._state_store = SocialStateStore()
        self._ensure_state_loaded()
        self.active_collaborations = _GLOBAL_ACTIVE_COLLABORATIONS
        self.social_graph = _GLOBAL_SOCIAL_GRAPH
        self.holon_service = HolonService()

    def _is_holon_runnable(self, holon_id: str) -> bool:
        """Return True when the Holon can be scheduled for collaboration."""
        try:
            return self.holon_service.is_active(holon_id)
        except Exception:
            return False

    @classmethod
    def reset_in_memory_cache(cls) -> None:
        """Reset process cache (useful for tests and root switches)."""
        global _GLOBAL_SOCIAL_GRAPH
        global _GLOBAL_COLLAB_STATE_ROOT
        _GLOBAL_ACTIVE_COLLABORATIONS.clear()
        _GLOBAL_SOCIAL_GRAPH = SocialGraph()
        _GLOBAL_COLLAB_STATE_ROOT = None

    def _ensure_state_loaded(self) -> None:
        """Load state from disk when process cache is empty or root changed."""
        global _GLOBAL_COLLAB_STATE_ROOT
        root_key = str(settings.holonpolis_root)
        if _GLOBAL_COLLAB_STATE_ROOT != root_key:
            self.reset_in_memory_cache()
            _GLOBAL_COLLAB_STATE_ROOT = root_key
            self._load_state_from_disk()
            return

        if not _GLOBAL_ACTIVE_COLLABORATIONS and not _GLOBAL_SOCIAL_GRAPH.relationships and not _GLOBAL_SOCIAL_GRAPH.reputations:
            self._load_state_from_disk()

    def _load_state_from_disk(self) -> None:
        payload = self._state_store.load_collaboration_state()
        self._load_active_collaborations(payload.get("active_collaborations", []))
        self._load_social_graph(payload.get("social_graph", {}))

    def persist_state(self) -> None:
        """Persist process-wide collaboration state to disk."""
        payload = {
            "active_collaborations": [self._task_to_dict(task) for task in _GLOBAL_ACTIVE_COLLABORATIONS.values()],
            "social_graph": self._social_graph_to_dict(_GLOBAL_SOCIAL_GRAPH),
            "updated_at": utc_now_iso(),
        }
        self._state_store.save_collaboration_state(payload)

    def _load_active_collaborations(self, tasks_data: List[Dict[str, Any]]) -> None:
        for item in tasks_data:
            task = self._task_from_dict(item)
            _GLOBAL_ACTIVE_COLLABORATIONS[task.task_id] = task

    def _load_social_graph(self, data: Dict[str, Any]) -> None:
        global _GLOBAL_SOCIAL_GRAPH
        graph = SocialGraph()
        for rel_data in data.get("relationships", []):
            rel_type_raw = str(rel_data.get("rel_type", RelationshipType.PEER.value))
            try:
                rel_type = RelationshipType(rel_type_raw)
            except ValueError:
                rel_type = RelationshipType.PEER
            relationship = SocialRelationship(
                relationship_id=str(rel_data.get("relationship_id", "")),
                source_holon=str(rel_data.get("source_holon", "")),
                target_holon=str(rel_data.get("target_holon", "")),
                rel_type=rel_type,
                strength=float(rel_data.get("strength", 0.5)),
                trust_score=float(rel_data.get("trust_score", 0.5)),
                created_at=str(rel_data.get("created_at", utc_now_iso())),
                last_interaction=rel_data.get("last_interaction"),
                interaction_count=int(rel_data.get("interaction_count", 0)),
                metadata=dict(rel_data.get("metadata", {})),
            )
            graph.add_relationship(relationship)

        for rep_data in data.get("reputations", []):
            rep = Reputation(
                holon_id=str(rep_data.get("holon_id", "")),
                overall_score=float(rep_data.get("overall_score", 0.5)),
                reliability=float(rep_data.get("reliability", 0.5)),
                competence=float(rep_data.get("competence", 0.5)),
                collaboration=float(rep_data.get("collaboration", 0.5)),
                innovation=float(rep_data.get("innovation", 0.5)),
                total_tasks=int(rep_data.get("total_tasks", 0)),
                successful_tasks=int(rep_data.get("successful_tasks", 0)),
                failed_tasks=int(rep_data.get("failed_tasks", 0)),
                collaboration_count=int(rep_data.get("collaboration_count", 0)),
                history=list(rep_data.get("history", [])),
            )
            graph.reputations[rep.holon_id] = rep

        _GLOBAL_SOCIAL_GRAPH = graph

    @staticmethod
    def _task_to_dict(task: CollaborationTask) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "state": task.state.name,
            "coordinator": task.coordinator,
            "participants": task.participants,
            "subtasks": [
                {
                    "subtask_id": subtask.subtask_id,
                    "name": subtask.name,
                    "description": subtask.description,
                    "assigned_to": subtask.assigned_to,
                    "status": subtask.status,
                    "result": CollaborationService._json_safe(subtask.result),
                    "deliverable": subtask.deliverable,
                }
                for subtask in task.subtasks
            ],
            "dependencies": task.dependencies,
            "deliverables": task.deliverables,
            "result": CollaborationService._json_safe(task.result),
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "deadline": task.deadline,
            "budget_tokens": task.budget_tokens,
            "used_tokens": task.used_tokens,
        }

    @staticmethod
    def _task_from_dict(data: Dict[str, Any]) -> CollaborationTask:
        state_raw = str(data.get("state", CollaborationState.FORMING.name))
        try:
            state = CollaborationState[state_raw]
        except KeyError:
            state = CollaborationState.FORMING

        task = CollaborationTask(
            task_id=str(data.get("task_id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            state=state,
            coordinator=str(data.get("coordinator", "")),
            participants=dict(data.get("participants", {})),
            dependencies=dict(data.get("dependencies", {})),
            deliverables=list(data.get("deliverables", [])),
            result=data.get("result"),
            created_at=str(data.get("created_at", utc_now_iso())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            deadline=data.get("deadline"),
            budget_tokens=int(data.get("budget_tokens", 100000)),
            used_tokens=int(data.get("used_tokens", 0)),
        )
        task.subtasks = [
            SubTask(
                subtask_id=str(item.get("subtask_id", "")),
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                assigned_to=item.get("assigned_to"),
                status=str(item.get("status", "pending")),
                result=item.get("result"),
                deliverable=item.get("deliverable"),
            )
            for item in data.get("subtasks", [])
            if isinstance(item, dict)
        ]
        return task

    @staticmethod
    def _social_graph_to_dict(graph: SocialGraph) -> Dict[str, Any]:
        relationships = []
        for rel in graph.relationships.values():
            relationships.append(
                {
                    "relationship_id": rel.relationship_id,
                    "source_holon": rel.source_holon,
                    "target_holon": rel.target_holon,
                    "rel_type": rel.rel_type.value,
                    "strength": rel.strength,
                    "trust_score": rel.trust_score,
                    "created_at": rel.created_at,
                    "last_interaction": rel.last_interaction,
                    "interaction_count": rel.interaction_count,
                    "metadata": rel.metadata,
                }
            )

        reputations = []
        for rep in graph.reputations.values():
            reputations.append(
                {
                    "holon_id": rep.holon_id,
                    "overall_score": rep.overall_score,
                    "reliability": rep.reliability,
                    "competence": rep.competence,
                    "collaboration": rep.collaboration,
                    "innovation": rep.innovation,
                    "total_tasks": rep.total_tasks,
                    "successful_tasks": rep.successful_tasks,
                    "failed_tasks": rep.failed_tasks,
                    "collaboration_count": rep.collaboration_count,
                    "history": rep.history,
                }
            )
        return {"relationships": relationships, "reputations": reputations}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Best-effort conversion of nested values to JSON-safe structures."""
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {str(k): CollaborationService._json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [CollaborationService._json_safe(item) for item in value]
            return str(value)

    async def create_collaboration(
        self,
        name: str,
        description: str,
        coordinator_id: str,
        participant_ids: List[str],
        task_structure: Dict[str, Any],
    ) -> CollaborationTask:
        """创建新的协作任务。

        Args:
            name: 协作名称
            description: 任务描述
            coordinator_id: 协调者 Holon ID
            participant_ids: 参与者 Holon IDs
            task_structure: 任务结构定义

        Returns:
            CollaborationTask
        """
        if not self._is_holon_runnable(coordinator_id):
            raise HolonUnavailableError(coordinator_id, self.holon_service.get_holon_status(coordinator_id), "coordinate")

        task_id = f"collab_{uuid.uuid4().hex[:12]}"

        # 创建协作任务
        task = CollaborationTask(
            task_id=task_id,
            name=name,
            description=description,
            state=CollaborationState.FORMING,
            coordinator=coordinator_id,
        )

        # 添加参与者
        for holon_id in participant_ids:
            if not self._is_holon_runnable(holon_id):
                logger.info("collaboration_skip_non_runnable_participant", task_id=task_id, holon_id=holon_id)
                continue
            task.participants[holon_id] = {
                "role": "worker",
                "status": "invited",
                "contribution": 0.0,
            }

        if coordinator_id not in task.participants:
            task.participants[coordinator_id] = {
                "role": "coordinator",
                "status": "invited",
                "contribution": 0.0,
            }

        # 设置协调者
        if coordinator_id in task.participants:
            task.participants[coordinator_id]["role"] = "coordinator"

        # 创建子任务
        for subtask_def in task_structure.get("subtasks", []):
            subtask = SubTask(
                subtask_id=f"{task_id}_sub_{uuid.uuid4().hex[:8]}",
                name=subtask_def["name"],
                description=subtask_def["description"],
            )
            task.subtasks.append(subtask)

        task.dependencies = task_structure.get("dependencies", {})

        self.active_collaborations[task_id] = task
        self.persist_state()

        logger.info(
            "collaboration_created",
            task_id=task_id,
            coordinator=coordinator_id,
            participants=len(participant_ids),
        )

        return task

    async def start_collaboration(self, task_id: str) -> bool:
        """启动协作任务。"""
        task = self.active_collaborations.get(task_id)
        if not task:
            return False

        task.state = CollaborationState.ACTIVE
        task.started_at = utc_now_iso()

        logger.info("collaboration_started", task_id=task_id)

        # 异步执行协作
        asyncio.create_task(self._execute_collaboration(task_id))
        self.persist_state()

        return True

    async def _execute_collaboration(self, task_id: str) -> None:
        """执行协作任务。"""
        task = self.active_collaborations.get(task_id)
        if not task:
            return

        try:
            # 1. 分配子任务
            await self._assign_subtasks(task)

            # 2. 按依赖顺序执行
            completed = set()
            failed = []

            while len(completed) < len(task.subtasks):
                # 找到可以执行的子任务（依赖已满足）
                ready = self._get_ready_subtasks(task, completed)

                if not ready:
                    if failed:
                        break  # 有失败的依赖，无法继续
                    await asyncio.sleep(0.1)
                    continue

                # 并行执行准备好的子任务
                results = await asyncio.gather(
                    *[self._execute_subtask(task, st) for st in ready],
                    return_exceptions=True,
                )

                for subtask, result in zip(ready, results):
                    if isinstance(result, Exception):
                        subtask.status = "failed"
                        failed.append(subtask.subtask_id)
                        logger.error(
                            "subtask_failed",
                            task_id=task_id,
                            subtask=subtask.subtask_id,
                            error=str(result),
                        )
                    else:
                        subtask.status = "completed"
                        subtask.result = result
                        completed.add(subtask.subtask_id)

            # 3. 汇总结果
            if len(completed) == len(task.subtasks):
                task.result = await self._aggregate_results(task)
                task.state = CollaborationState.COMPLETED
                logger.info("collaboration_completed", task_id=task_id)
            else:
                task.state = CollaborationState.FAILED
                logger.error(
                    "collaboration_failed",
                    task_id=task_id,
                    completed=len(completed),
                    failed=len(failed),
                )

        except Exception as e:
            task.state = CollaborationState.FAILED
            logger.error("collaboration_error", task_id=task_id, error=str(e))

        task.completed_at = utc_now_iso()
        self.persist_state()

    async def _assign_subtasks(self, task: CollaborationTask) -> None:
        """分配子任务给参与者。"""
        workers = [
            hid for hid, info in task.participants.items()
            if info["role"] == "worker" and self._is_holon_runnable(hid)
        ]

        if not workers:
            return

        # 简单轮询分配
        for i, subtask in enumerate(task.subtasks):
            worker_id = workers[i % len(workers)]
            subtask.assigned_to = worker_id

            logger.debug(
                "subtask_assigned",
                task_id=task.task_id,
                subtask=subtask.subtask_id,
                worker=worker_id,
            )
        self.persist_state()

    def _get_ready_subtasks(
        self,
        task: CollaborationTask,
        completed: set,
    ) -> List[SubTask]:
        """获取可以执行的子任务（依赖已满足）。"""
        ready = []

        for subtask in task.subtasks:
            if subtask.status != "pending":
                continue

            deps = task.dependencies.get(subtask.subtask_id, [])
            if all(d in completed for d in deps):
                ready.append(subtask)

        return ready

    async def _execute_subtask(
        self,
        task: CollaborationTask,
        subtask: SubTask,
    ) -> Any:
        """执行单个子任务。"""
        if not subtask.assigned_to:
            raise ValueError(f"Subtask {subtask.subtask_id} not assigned")

        # 获取 Holon runtime
        holon = self._get_holon_runtime(subtask.assigned_to)

        # 构建任务描述
        prompt = f"""Execute this subtask as part of a larger collaboration:

Collaboration: {task.name}
Overall Goal: {task.description}

Your Subtask: {subtask.name}
Description: {subtask.description}

Execute the subtask and return the result."""

        # 执行
        result = await holon.chat(prompt)

        # 更新贡献
        task.participants[subtask.assigned_to]["contribution"] += 1.0 / len(task.subtasks)
        self.persist_state()

        return result.get("content", "")

    async def _aggregate_results(self, task: CollaborationTask) -> Dict[str, Any]:
        """汇总所有子任务的结果。"""
        # 由协调者进行汇总
        coordinator = self._get_holon_runtime(task.coordinator)

        # 构建汇总提示
        results_text = "\n\n".join([
            f"Subtask: {st.name}\nResult: {st.result}"
            for st in task.subtasks if st.result
        ])

        prompt = f"""Aggregate these subtask results into a cohesive final output:

Collaboration: {task.name}
Goal: {task.description}

Subtask Results:
{results_text}

Provide a comprehensive final result."""

        result = await coordinator.chat(prompt)

        return {
            "final_output": result.get("content", ""),
            "subtask_results": {st.name: st.result for st in task.subtasks},
            "participants": task.participants,
        }

    def _get_holon_runtime(self, holon_id: str) -> HolonRuntime:
        """获取 Holon runtime 实例。"""
        self.holon_service.assert_runnable(holon_id, action="collaborate")
        blueprint = self.holon_service.get_blueprint(holon_id)
        return HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    async def find_collaborators(
        self,
        holon_id: str,
        skill_needed: str,
        min_reputation: float = 0.3,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """为指定 Holon 寻找合适的协作者。"""
        # 从社会图中查找
        candidates = self.social_graph.find_collaborators(
            holon_id=holon_id,
            skill_required=skill_needed,
            min_trust=min_reputation,
            top_k=top_k,
        )
        candidates = [
            (hid, score)
            for hid, score in candidates
            if self._is_holon_runnable(hid)
        ]

        if len(candidates) < top_k:
            # 从 Genesis 查找更多 Holon
            all_holons = self.holon_service.list_holons()
            existing = {hid for hid, _ in candidates}
            existing.add(holon_id)

            for holon_data in all_holons:
                hid = holon_data["holon_id"]
                if hid in existing:
                    continue
                if str(holon_data.get("status") or "active") != "active":
                    continue

                # 检查技能匹配
                if skill_needed.lower() in holon_data.get("purpose", "").lower():
                    reputation = self.social_graph.get_reputation(hid)
                    if reputation.overall_score >= min_reputation:
                        candidates.append((hid, reputation.overall_score * 0.5))

        return candidates[:top_k]

    def register_relationship(self, relationship: SocialRelationship) -> None:
        """Register a relationship in the social graph and persist."""
        self.social_graph.add_relationship(relationship)
        self.persist_state()

    def get_collaboration_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取协作任务状态。"""
        task = self.active_collaborations.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "name": task.name,
            "state": task.state.name,
            "participants": len(task.participants),
            "subtasks_total": len(task.subtasks),
            "subtasks_completed": sum(1 for st in task.subtasks if st.status == "completed"),
            "progress": sum(1 for st in task.subtasks if st.status == "completed") / len(task.subtasks) if task.subtasks else 0,
        }
