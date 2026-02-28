"""Holon Social Layer - 多智能体社会的核心定义。

定义 Holon 之间的:
1. 协作关系 (Collaboration)
2. 竞争/市场机制 (Competition/Market)
3. 社会网络 (Social Network)
4. 信任与声誉 (Trust & Reputation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class RelationshipType(Enum):
    """Holon 之间的关系类型。"""
    PARENT = "parent"           # 父-子（演化关系）
    COLLABORATOR = "collaborator"  # 协作关系
    COMPETITOR = "competitor"   # 竞争关系
    MENTOR = "mentor"          # 导师-学徒
    CLIENT = "client"          # 服务提供者-客户
    PEER = "peer"              # 对等关系


class CollaborationState(Enum):
    """协作状态。"""
    FORMING = auto()      # 正在组建
    ACTIVE = auto()       # 进行中
    PAUSED = auto()       # 暂停
    COMPLETED = auto()    # 完成
    FAILED = auto()       # 失败
    DISSOLVED = auto()    # 解散


@dataclass
class SocialRelationship:
    """Holon 之间的社会关系。"""
    relationship_id: str
    source_holon: str
    target_holon: str
    rel_type: RelationshipType
    strength: float = 0.5       # 关系强度 0-1
    trust_score: float = 0.5    # 信任分数 0-1
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_interaction: Optional[str] = None
    interaction_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_interaction(self, outcome: str, quality: float = 0.5) -> None:
        """记录一次交互并更新关系。"""
        self.interaction_count += 1
        self.last_interaction = datetime.utcnow().isoformat()

        # 根据交互结果更新信任分数
        if outcome == "success":
            self.trust_score = min(1.0, self.trust_score + quality * 0.1)
            self.strength = min(1.0, self.strength + 0.05)
        elif outcome == "failure":
            self.trust_score = max(0.0, self.trust_score - 0.1)
            self.strength = max(0.0, self.strength - 0.05)


@dataclass
class CollaborationTask:
    """协作任务定义。"""
    task_id: str
    name: str
    description: str
    state: CollaborationState

    # 参与者
    coordinator: str                    # 协调者 Holon ID
    participants: Dict[str, Dict] = field(default_factory=dict)  # holon_id -> {role, status, contribution}

    # 任务结构
    subtasks: List[SubTask] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # subtask_id -> [dependency_ids]

    # 结果
    deliverables: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None

    # 时间
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    deadline: Optional[str] = None

    # 资源
    budget_tokens: int = 100000
    used_tokens: int = 0


@dataclass
class SubTask:
    """子任务。"""
    subtask_id: str
    name: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    deliverable: Optional[str] = None


@dataclass
class Reputation:
    """Holon 的声誉记录。"""
    holon_id: str

    # 总体声誉
    overall_score: float = 0.5  # 0-1

    # 维度评分
    reliability: float = 0.5      # 可靠性
    competence: float = 0.5       # 能力
    collaboration: float = 0.5    # 协作性
    innovation: float = 0.5       # 创新性

    # 历史统计
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    collaboration_count: int = 0

    # 时间序列（用于趋势分析）
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update(self, event_type: str, outcome: str, rating: float = 0.5) -> None:
        """更新声誉。"""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "outcome": outcome,
            "rating": rating,
        }
        self.history.append(record)

        if event_type == "task":
            self.total_tasks += 1
            if outcome == "success":
                self.successful_tasks += 1
                self.reliability = min(1.0, self.reliability + 0.02)
            else:
                self.failed_tasks += 1
                self.reliability = max(0.0, self.reliability - 0.05)

        elif event_type == "collaboration":
            self.collaboration_count += 1
            if outcome == "success":
                self.collaboration = min(1.0, self.collaboration + 0.03)

        # 重新计算总体声誉
        self.overall_score = (
            self.reliability * 0.3 +
            self.competence * 0.3 +
            self.collaboration * 0.25 +
            self.innovation * 0.15
        )


@dataclass
class MarketOffer:
    """技能市场上的报价。"""
    offer_id: str
    holon_id: str
    skill_name: str
    skill_description: str

    # 定价
    price_per_use: float = 0.0      # 每次使用的价格（token 数）
    price_per_month: float = 0.0    # 包月价格

    # 质量承诺
    success_rate: float = 0.0       # 宣称成功率
    avg_latency_ms: int = 0         # 平均响应时间

    # 市场数据
    usage_count: int = 0
    rating: float = 0.0             # 用户评价
    reviews: List[Dict] = field(default_factory=list)

    # 状态
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CompetitionResult:
    """竞争/评估结果。"""
    competition_id: str
    task_description: str
    participants: List[str]  # holon_ids

    # 评分维度
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)  # holon_id -> {accuracy, speed, cost, quality}

    # 排名
    ranking: List[str] = field(default_factory=list)  # 按总分排序的 holon_ids

    # 奖励/惩罚
    winner: Optional[str] = None
    rewards: Dict[str, float] = field(default_factory=dict)  # holon_id -> token_reward

    completed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SocialGraph:
    """Holon 社会网络图。

    管理所有 Holon 之间的关系网络，支持:
    - 关系查询
    - 信任传播
    - 社区发现
    - 影响力分析
    """

    def __init__(self):
        self.relationships: Dict[str, SocialRelationship] = {}
        self.reputations: Dict[str, Reputation] = {}
        self._index_by_holon: Dict[str, Set[str]] = {}  # holon_id -> relationship_ids

    def add_relationship(self, rel: SocialRelationship) -> None:
        """添加关系。"""
        self.relationships[rel.relationship_id] = rel

        # 更新索引
        for holon_id in [rel.source_holon, rel.target_holon]:
            if holon_id not in self._index_by_holon:
                self._index_by_holon[holon_id] = set()
            self._index_by_holon[holon_id].add(rel.relationship_id)

    def get_relationships(
        self,
        holon_id: str,
        rel_type: Optional[RelationshipType] = None,
        min_strength: float = 0.0,
    ) -> List[SocialRelationship]:
        """获取 Holon 的关系。"""
        rel_ids = self._index_by_holon.get(holon_id, set())
        result = []

        for rel_id in rel_ids:
            rel = self.relationships.get(rel_id)
            if not rel:
                continue

            if rel_type and rel.rel_type != rel_type:
                continue

            if rel.strength < min_strength:
                continue

            result.append(rel)

        return sorted(result, key=lambda r: r.strength, reverse=True)

    def get_reputation(self, holon_id: str) -> Reputation:
        """获取或创建声誉记录。"""
        if holon_id not in self.reputations:
            self.reputations[holon_id] = Reputation(holon_id=holon_id)
        return self.reputations[holon_id]

    def find_collaborators(
        self,
        holon_id: str,
        skill_required: Optional[str] = None,
        min_trust: float = 0.3,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """寻找潜在的协作者。

        Returns:
            List of (holon_id, compatibility_score)
        """
        # 获取已有的关系
        relationships = self.get_relationships(holon_id, min_strength=min_trust)

        candidates = []
        for rel in relationships:
            other = rel.target_holon if rel.source_holon == holon_id else rel.source_holon
            reputation = self.get_reputation(other)

            # 计算兼容性分数
            score = (
                rel.trust_score * 0.3 +
                reputation.overall_score * 0.3 +
                reputation.collaboration * 0.2 +
                rel.strength * 0.2
            )

            candidates.append((other, score))

        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def propagate_trust(
        self,
        source_holon: str,
        target_holon: str,
        max_hops: int = 2,
    ) -> float:
        """通过社交网络传播计算间接信任。

        使用类似 PageRank 的算法计算间接信任。
        """
        if max_hops <= 0:
            return 0.0

        # 直接信任
        direct_rels = [
            r for r in self.get_relationships(source_holon)
            if r.target_holon == target_holon or r.source_holon == target_holon
        ]

        if direct_rels:
            return max(r.trust_score for r in direct_rels)

        # 间接信任（通过共同连接）
        indirect_trust = 0.0
        source_rels = self.get_relationships(source_holon)

        for rel in source_rels:
            intermediate = rel.target_holon if rel.source_holon == source_holon else rel.source_holon
            intermediate_trust = rel.trust_score

            # 递归计算
            next_level_trust = self.propagate_trust(
                intermediate, target_holon, max_hops - 1
            )

            # 信任衰减
            indirect_trust = max(indirect_trust, intermediate_trust * next_level_trust * 0.5)

        return indirect_trust
