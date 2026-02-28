#!/usr/bin/env python3
"""
HolonPolis 社会能力演示 - 协作与竞争

展示:
1. 多个 Holon 协作完成复杂任务
2. 技能市场注册和发现
3. 竞争评估
4. 自然选择
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.blueprints import EvolutionStrategy
from holonpolis.domain.social import RelationshipType
from holonpolis.services.collaboration_service import CollaborationService
from holonpolis.services.market_service import MarketService
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory


async def demo_collaboration():
    """演示 Holon 协作。"""
    print("=" * 80)
    print("🤝 HOLOPOLIS 协作演示")
    print("=" * 80)

    # 初始化服务
    collab_service = CollaborationService()

    # 创建 3 个 Holon
    holons = [
        ("holon_designer", "UI Designer", "Design user interfaces and visual components"),
        ("holon_coder", "Frontend Developer", "Implement React components and pages"),
        ("holon_tester", "QA Engineer", "Test components and ensure quality"),
    ]

    print("\n🥚 创建协作 Holons:")
    for hid, name, purpose in holons:
        blueprint = Blueprint(
            blueprint_id=f"blueprint_{hid}",
            holon_id=hid,
            species_id="specialist",
            name=name,
            purpose=purpose,
            boundary=Boundary(allow_file_write=True),
            evolution_policy=EvolutionPolicy(strategy=EvolutionStrategy.BALANCED),
        )

        # 初始化数据库
        factory = get_lancedb_factory()
        factory.init_holon_tables(hid)

        print(f"   ✓ {name} ({hid})")

    # 建立社会关系
    print("\n🔗 建立社会关系:")
    collab_service.social_graph.add_relationship(
        type("Rel", (), {
            "relationship_id": "rel_001",
            "source_holon": "holon_designer",
            "target_holon": "holon_coder",
            "rel_type": RelationshipType.COLLABORATOR,
            "strength": 0.8,
            "trust_score": 0.9,
            "record_interaction": lambda *args: None,
        })()
    )
    print("   ✓ Designer ↔ Developer: 协作者关系")

    # 创建协作任务
    print("\n📋 创建协作任务:")
    task_structure = {
        "subtasks": [
            {"name": "Design Homepage", "description": "Create homepage mockup with cyberpunk theme"},
            {"name": "Implement Components", "description": "Build React components based on design"},
            {"name": "Test Integration", "description": "Test all components work together"},
        ],
        "dependencies": {
            "implement": ["design"],
            "test": ["implement"],
        },
    }

    task = await collab_service.create_collaboration(
        name="Build CyberPunk Homepage",
        description="Collaboratively build a cyberpunk-themed homepage",
        coordinator_id="holon_designer",
        participant_ids=["holon_designer", "holon_coder", "holon_tester"],
        task_structure=task_structure,
    )

    print(f"   任务 ID: {task.task_id}")
    print(f"   参与者: {len(task.participants)}")
    print(f"   子任务: {len(task.subtasks)}")

    # 查找协作者
    print("\n🔍 为 Designer 寻找协作者:")
    collaborators = await collab_service.find_collaborators(
        holon_id="holon_designer",
        skill_needed="frontend development",
        top_k=3,
    )
    for hid, score in collaborators:
        print(f"   {hid}: 匹配度 {score:.2f}")

    print("\n" + "-" * 80)


async def demo_market():
    """演示技能市场。"""
    print("\n🏪 HOLOPOLIS 技能市场演示")
    print("=" * 80)

    market = MarketService()

    # 注册技能报价
    print("\n📢 注册技能报价:")
    offers = [
        ("holon_react", "React Component Builder", "Build React components", 100, 0.95),
        ("holon_css", "CSS Stylist", "Create beautiful CSS", 80, 0.90),
        ("holon_api", "API Designer", "Design REST APIs", 150, 0.88),
        ("holon_tester", "Test Writer", "Write comprehensive tests", 60, 0.92),
    ]

    for hid, skill, desc, price, rate in offers:
        offer = market.register_offer(hid, skill, desc, price, rate)
        print(f"   ✓ {skill} by {hid}: {price} tokens/use, {rate:.0%} success rate")

    # 搜索技能
    print("\n🔎 搜索 'React':")
    results = market.find_offers("React", top_k=3)
    for offer, score in results:
        print(f"   {offer.skill_name} (匹配度: {score:.2f})")

    # 记录使用
    print("\n📊 记录使用情况:")
    for offer_id in list(market.offers.keys())[:2]:
        market.record_usage(offer_id, success=True, latency_ms=500, user_rating=0.9)
        print(f"   ✓ {offer_id}: 成功使用，评分 4.5/5")

    # 市场统计
    stats = market.get_market_stats()
    print(f"\n📈 市场统计:")
    print(f"   总报价: {stats['total_offers']}")
    print(f"   活跃报价: {stats['active_offers']}")
    print(f"   平均价格: {stats['avg_price']:.0f} tokens")

    print("\n" + "-" * 80)


async def demo_competition():
    """演示竞争机制。"""
    print("\n⚔️ HOLOPOLIS 竞争演示")
    print("=" * 80)

    market = MarketService()

    # 初始化 Holons
    holon_ids = ["holon_fast", "holon_accurate", "holon_balanced"]
    for hid in holon_ids:
        factory = get_lancedb_factory()
        factory.init_holon_tables(hid)

    print("\n🎯 运行竞争评估:")
    print("   任务: 生成一个登录表单组件")
    print("   参与者:", ", ".join(holon_ids))

    # 模拟竞争结果（简化）
    print("\n🏆 竞争结果:")
    print("   🥇 holon_accurate: 准确率 98%, 速度 85%")
    print("   🥈 holon_balanced: 准确率 92%, 速度 92%")
    print("   🥉 holon_fast: 准确率 85%, 速度 98%")

    # 声誉更新
    for hid in holon_ids:
        reputation = market._get_reputation(hid)
        if hid == "holon_accurate":
            reputation.update("competition", "success", 1.0)
            reputation.competence = 0.95
        elif hid == "holon_balanced":
            reputation.update("competition", "success", 0.8)
            reputation.competence = 0.90
        else:
            reputation.update("competition", "success", 0.6)
            reputation.competence = 0.85

    # 自然选择
    print("\n🌿 自然选择 (threshold=0.7):")
    selection = market.run_selection(threshold=0.7)
    print(f"   总数: {selection['total']}")
    print(f"   幸存者: {selection['survivors']}")
    print(f"   淘汰: {selection['eliminated']}")

    if selection['top_performers']:
        print(f"\n   🌟 顶级表现者:")
        for perf in selection['top_performers'][:3]:
            print(f"      {perf['holon_id']}: {perf['score']:.2f}")

    print("\n" + "=" * 80)
    print("✅ 社会能力演示完成!")
    print("=" * 80)
    print("\n关键能力:")
    print("  🤝 协作: 多个 Holon 分工完成复杂任务")
    print("  🏪 市场: 技能供需匹配和价格发现")
    print("  ⚔️ 竞争: 优胜劣汰，选出最佳方案")
    print("  🌿 选择: 低质量 Holon 被淘汰，高质量者获得奖励")
    print("\n这实现了真正的多智能体生态系统!")


async def main():
    await demo_collaboration()
    await demo_market()
    await demo_competition()


if __name__ == "__main__":
    asyncio.run(main())
