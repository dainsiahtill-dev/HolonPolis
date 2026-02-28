"""Market Service - 技能市场和竞争机制。

核心功能:
1. 技能交易市场 (Marketplace)
2. 竞争评估 (Competition)
3. 优胜劣汰 (Selection)
4. 价格发现 (Price Discovery)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from holonpolis.config import settings
from holonpolis.domain.social import CompetitionResult, MarketOffer, Reputation
from holonpolis.services.holon_service import HolonService
from holonpolis.services.social_state_store import SocialStateStore

logger = structlog.get_logger()

# Process-wide state to keep market continuity across service instances.
_GLOBAL_OFFERS: Dict[str, MarketOffer] = {}
_GLOBAL_REPUTATION_REGISTRY: Dict[str, Reputation] = {}
_GLOBAL_COMPETITION_HISTORY: List[CompetitionResult] = []
_GLOBAL_MARKET_STATE_ROOT: Optional[str] = None


class MarketService:
    """技能市场服务。

    管理 Holon 技能的供需关系和定价。
    """

    def __init__(self):
        self.offers = _GLOBAL_OFFERS
        self.reputation_registry = _GLOBAL_REPUTATION_REGISTRY
        self.competition_history = _GLOBAL_COMPETITION_HISTORY
        self.holon_service = HolonService()
        self._state_store = SocialStateStore()
        self._ensure_state_loaded()

    @classmethod
    def reset_in_memory_cache(cls) -> None:
        """Reset process cache (useful for tests and root switches)."""
        global _GLOBAL_MARKET_STATE_ROOT
        _GLOBAL_OFFERS.clear()
        _GLOBAL_REPUTATION_REGISTRY.clear()
        _GLOBAL_COMPETITION_HISTORY.clear()
        _GLOBAL_MARKET_STATE_ROOT = None

    def _ensure_state_loaded(self) -> None:
        """Load state from disk when process cache is empty or root changed."""
        global _GLOBAL_MARKET_STATE_ROOT
        root_key = str(settings.holonpolis_root)
        if _GLOBAL_MARKET_STATE_ROOT != root_key:
            self.reset_in_memory_cache()
            _GLOBAL_MARKET_STATE_ROOT = root_key
            self._load_state_from_disk()
            return

        if not self.offers and not self.reputation_registry and not self.competition_history:
            self._load_state_from_disk()

    def _load_state_from_disk(self) -> None:
        payload = self._state_store.load_market_state()
        offers_data = payload.get("offers", [])
        reputations_data = payload.get("reputations", [])
        history_data = payload.get("competition_history", [])

        for item in offers_data:
            offer = self._offer_from_dict(item)
            self.offers[offer.offer_id] = offer

        for item in reputations_data:
            rep = self._reputation_from_dict(item)
            self.reputation_registry[rep.holon_id] = rep

        for item in history_data:
            self.competition_history.append(self._competition_from_dict(item))

    def persist_state(self) -> None:
        """Persist process-wide market state to disk."""
        payload = {
            "offers": [self._offer_to_dict(item) for item in self.offers.values()],
            "reputations": [self._reputation_to_dict(item) for item in self.reputation_registry.values()],
            "competition_history": [self._competition_to_dict(item) for item in self.competition_history],
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._state_store.save_market_state(payload)

    @staticmethod
    def _offer_to_dict(offer: MarketOffer) -> Dict[str, Any]:
        return {
            "offer_id": offer.offer_id,
            "holon_id": offer.holon_id,
            "skill_name": offer.skill_name,
            "skill_description": offer.skill_description,
            "price_per_use": offer.price_per_use,
            "price_per_month": offer.price_per_month,
            "success_rate": offer.success_rate,
            "avg_latency_ms": offer.avg_latency_ms,
            "usage_count": offer.usage_count,
            "rating": offer.rating,
            "reviews": offer.reviews,
            "is_active": offer.is_active,
            "created_at": offer.created_at,
        }

    @staticmethod
    def _offer_from_dict(data: Dict[str, Any]) -> MarketOffer:
        return MarketOffer(
            offer_id=str(data.get("offer_id", "")),
            holon_id=str(data.get("holon_id", "")),
            skill_name=str(data.get("skill_name", "")),
            skill_description=str(data.get("skill_description", "")),
            price_per_use=float(data.get("price_per_use", 0.0)),
            price_per_month=float(data.get("price_per_month", 0.0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_latency_ms=int(data.get("avg_latency_ms", 0)),
            usage_count=int(data.get("usage_count", 0)),
            rating=float(data.get("rating", 0.0)),
            reviews=list(data.get("reviews", [])),
            is_active=bool(data.get("is_active", True)),
            created_at=str(data.get("created_at", datetime.utcnow().isoformat())),
        )

    @staticmethod
    def _reputation_to_dict(rep: Reputation) -> Dict[str, Any]:
        return {
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

    @staticmethod
    def _reputation_from_dict(data: Dict[str, Any]) -> Reputation:
        return Reputation(
            holon_id=str(data.get("holon_id", "")),
            overall_score=float(data.get("overall_score", 0.5)),
            reliability=float(data.get("reliability", 0.5)),
            competence=float(data.get("competence", 0.5)),
            collaboration=float(data.get("collaboration", 0.5)),
            innovation=float(data.get("innovation", 0.5)),
            total_tasks=int(data.get("total_tasks", 0)),
            successful_tasks=int(data.get("successful_tasks", 0)),
            failed_tasks=int(data.get("failed_tasks", 0)),
            collaboration_count=int(data.get("collaboration_count", 0)),
            history=list(data.get("history", [])),
        )

    @staticmethod
    def _competition_to_dict(item: CompetitionResult) -> Dict[str, Any]:
        return {
            "competition_id": item.competition_id,
            "task_description": item.task_description,
            "participants": item.participants,
            "scores": item.scores,
            "ranking": item.ranking,
            "winner": item.winner,
            "rewards": item.rewards,
            "completed_at": item.completed_at,
        }

    @staticmethod
    def _competition_from_dict(data: Dict[str, Any]) -> CompetitionResult:
        return CompetitionResult(
            competition_id=str(data.get("competition_id", "")),
            task_description=str(data.get("task_description", "")),
            participants=list(data.get("participants", [])),
            scores=dict(data.get("scores", {})),
            ranking=list(data.get("ranking", [])),
            winner=data.get("winner"),
            rewards=dict(data.get("rewards", {})),
            completed_at=str(data.get("completed_at", datetime.utcnow().isoformat())),
        )

    # ========== 市场管理 ==========

    def register_offer(
        self,
        holon_id: str,
        skill_name: str,
        skill_description: str,
        price_per_use: float = 0.0,
        success_rate: float = 0.0,
    ) -> MarketOffer:
        """注册技能到市场。"""
        offer_id = f"offer_{uuid.uuid4().hex[:12]}"

        offer = MarketOffer(
            offer_id=offer_id,
            holon_id=holon_id,
            skill_name=skill_name,
            skill_description=skill_description,
            price_per_use=price_per_use,
            success_rate=success_rate,
        )

        self.offers[offer_id] = offer
        self.persist_state()

        logger.info(
            "offer_registered",
            offer_id=offer_id,
            holon_id=holon_id,
            skill=skill_name,
            price=price_per_use,
        )

        return offer

    def find_offers(
        self,
        skill_query: str,
        max_price: Optional[float] = None,
        min_rating: float = 0.0,
        top_k: int = 5,
    ) -> List[Tuple[MarketOffer, float]]:
        """搜索技能市场。

        Returns:
            List of (offer, match_score)
        """
        results = []

        for offer in self.offers.values():
            if not offer.is_active:
                continue

            # 价格筛选
            if max_price is not None and offer.price_per_use > max_price:
                continue

            # 评分筛选
            if offer.rating < min_rating:
                continue

            # 计算匹配度
            match_score = self._calculate_match(offer, skill_query)
            if match_score > 0.3:  # 最小匹配阈值
                results.append((offer, match_score))

        # 按综合分数排序 (匹配度 * 评分 / 价格)
        def ranking_key(item):
            offer, match = item
            price_factor = 1.0 / (1 + offer.price_per_use / 1000)  # 价格越低越好
            rating_factor = (offer.rating + 0.1) / 1.1  # 评分越高越好
            return match * rating_factor * price_factor

        results.sort(key=ranking_key, reverse=True)
        return results[:top_k]

    def _calculate_match(self, offer: MarketOffer, query: str) -> float:
        """计算技能与查询的匹配度。"""
        query_lower = query.lower()
        skill_lower = offer.skill_name.lower()
        desc_lower = offer.skill_description.lower()

        # 名称匹配
        if query_lower in skill_lower:
            return 1.0
        if skill_lower in query_lower:
            return 0.9

        # 描述匹配
        if query_lower in desc_lower:
            return 0.7

        # 关键词匹配
        query_words = set(query_lower.split())
        desc_words = set(desc_lower.split())
        overlap = len(query_words & desc_words)
        if overlap > 0:
            return 0.5 + 0.3 * (overlap / len(query_words))

        return 0.0

    def record_usage(
        self,
        offer_id: str,
        success: bool,
        latency_ms: int,
        user_rating: Optional[float] = None,
    ) -> None:
        """记录技能使用情况，更新市场数据。"""
        offer = self.offers.get(offer_id)
        if not offer:
            return

        offer.usage_count += 1

        # 更新平均延迟
        if offer.avg_latency_ms == 0:
            offer.avg_latency_ms = latency_ms
        else:
            offer.avg_latency_ms = int(
                (offer.avg_latency_ms * (offer.usage_count - 1) + latency_ms)
                / offer.usage_count
            )

        # 更新成功率
        current_success = offer.success_rate * (offer.usage_count - 1)
        if success:
            current_success += 1
        offer.success_rate = current_success / offer.usage_count

        # 更新评分
        if user_rating is not None:
            if offer.rating == 0:
                offer.rating = user_rating
            else:
                offer.rating = (offer.rating * (offer.usage_count - 1) + user_rating) / offer.usage_count

            offer.reviews.append({
                "timestamp": datetime.utcnow().isoformat(),
                "rating": user_rating,
                "success": success,
            })

        logger.debug(
            "usage_recorded",
            offer_id=offer_id,
            success=success,
            new_rating=offer.rating,
        )
        self.persist_state()

    # ========== 竞争机制 ==========

    async def run_competition(
        self,
        task_description: str,
        evaluation_criteria: Dict[str, float],  # 维度 -> 权重
        participant_ids: List[str],
        test_cases: List[Dict],
    ) -> CompetitionResult:
        """运行竞争评估。

        多个 Holon 竞争同一任务，评估最优者。
        """
        competition_id = f"comp_{uuid.uuid4().hex[:12]}"

        logger.info(
            "competition_started",
            competition_id=competition_id,
            participants=len(participant_ids),
            task=task_description[:50],
        )

        # 每个参与者执行任务
        scores: Dict[str, Dict[str, float]] = {}

        for holon_id in participant_ids:
            holon = self._get_holon(holon_id)
            holon_scores = await self._evaluate_holon(
                holon, task_description, test_cases, evaluation_criteria
            )
            scores[holon_id] = holon_scores

        # 计算总分并排名
        total_scores = {
            hid: sum(s * evaluation_criteria.get(dim, 1.0) for dim, s in holon_scores.items())
            for hid, holon_scores in scores.items()
        }

        ranking = sorted(total_scores.keys(), key=lambda x: total_scores[x], reverse=True)

        # 创建结果
        result = CompetitionResult(
            competition_id=competition_id,
            task_description=task_description,
            participants=participant_ids,
            scores=scores,
            ranking=ranking,
            winner=ranking[0] if ranking else None,
        )

        # 分配奖励
        if result.winner:
            result.rewards[result.winner] = 1000.0  # 基础奖励
            for i, hid in enumerate(ranking[1:3], 2):
                result.rewards[hid] = 500.0 / i  # 递减奖励

        self.competition_history.append(result)

        # 更新声誉
        for holon_id in participant_ids:
            reputation = self._get_reputation(holon_id)
            rank = ranking.index(holon_id) + 1 if holon_id in ranking else len(ranking)

            if rank == 1:
                reputation.update("competition", "success", 1.0)
            elif rank <= 3:
                reputation.update("competition", "success", 0.7)
            else:
                reputation.update("competition", "failure", 0.3)

        logger.info(
            "competition_completed",
            competition_id=competition_id,
            winner=result.winner,
            top_score=total_scores.get(result.winner, 0) if result.winner else 0,
        )
        self.persist_state()

        return result

    async def _evaluate_holon(
        self,
        holon,
        task_description: str,
        test_cases: List[Dict],
        criteria: Dict[str, float],
    ) -> Dict[str, float]:
        """评估单个 Holon 的表现。"""
        scores = {dim: 0.5 for dim in criteria.keys()}

        start_time = datetime.utcnow()

        try:
            # 执行测试用例
            results = []
            for tc in test_cases:
                result = await holon.chat(tc["input"])
                results.append(result)

            # 准确率
            correct = sum(
                1 for r, tc in zip(results, test_cases)
                if self._check_result(r, tc["expected"])
            )
            scores["accuracy"] = correct / len(test_cases) if test_cases else 0.5

            # 速度
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            scores["speed"] = max(0.0, 1.0 - elapsed / 60)  # 60秒内完成得满分

            # 成本 (token 使用)
            scores["cost"] = 0.5  # 简化计算

            # 质量
            scores["quality"] = scores["accuracy"] * 0.8 + 0.2  # 基础质量分

        except Exception as e:
            logger.error("evaluation_failed", holon_id=holon.holon_id, error=str(e))
            scores = {dim: 0.0 for dim in criteria.keys()}

        return scores

    def _check_result(self, actual: Dict, expected: Any) -> bool:
        """检查结果是否符合预期。"""
        content = actual.get("content", "")
        if isinstance(expected, str):
            return expected.lower() in content.lower()
        return False

    # ========== 优胜劣汰 ==========

    def run_selection(self, threshold: float = 0.3) -> Dict[str, Any]:
        """执行自然选择，淘汰低质量 Holon。

        Returns:
            选择结果统计
        """
        all_holons = self.holon_service.list_holons()

        survivors = []
        eliminated = []

        for holon_data in all_holons:
            holon_id = holon_data["holon_id"]
            reputation = self._get_reputation(holon_id)

            # 计算生存分数
            survival_score = (
                reputation.overall_score * 0.4 +
                reputation.reliability * 0.3 +
                (reputation.successful_tasks / max(1, reputation.total_tasks)) * 0.3
            )

            if survival_score >= threshold:
                survivors.append({
                    "holon_id": holon_id,
                    "score": survival_score,
                    "reputation": reputation,
                })
            else:
                eliminated.append({
                    "holon_id": holon_id,
                    "score": survival_score,
                    "reason": "low_reputation",
                })

        # 对幸存者排序
        survivors.sort(key=lambda x: x["score"], reverse=True)

        # 更新市场状态
        for e in eliminated:
            # 停用其技能报价
            for offer in self.offers.values():
                if offer.holon_id == e["holon_id"]:
                    offer.is_active = False

        result = {
            "total": len(all_holons),
            "survivors": len(survivors),
            "eliminated": len(eliminated),
            "top_performers": survivors[:5],
            "eliminated_list": eliminated,
            "selection_pressure": 1.0 - threshold,
        }

        logger.info(
            "selection_completed",
            total=result["total"],
            survivors=result["survivors"],
            eliminated=result["eliminated"],
        )
        self.persist_state()

        return result

    def get_market_stats(self) -> Dict[str, Any]:
        """获取市场统计信息。"""
        active_offers = [o for o in self.offers.values() if o.is_active]

        return {
            "total_offers": len(self.offers),
            "active_offers": len(active_offers),
            "total_usage": sum(o.usage_count for o in self.offers.values()),
            "avg_price": sum(o.price_per_use for o in active_offers) / len(active_offers) if active_offers else 0,
            "avg_rating": sum(o.rating for o in active_offers) / len(active_offers) if active_offers else 0,
            "total_holons": len(self.reputation_registry),
            "competitions_held": len(self.competition_history),
        }

    def _get_holon(self, holon_id: str):
        """获取 Holon runtime。"""
        from holonpolis.runtime.holon_runtime import HolonRuntime
        blueprint = self.holon_service.get_blueprint(holon_id)
        return HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    def _get_reputation(self, holon_id: str) -> Reputation:
        """获取或创建声誉记录。"""
        if holon_id not in self.reputation_registry:
            self.reputation_registry[holon_id] = Reputation(holon_id=holon_id)
            self.persist_state()
        return self.reputation_registry[holon_id]
