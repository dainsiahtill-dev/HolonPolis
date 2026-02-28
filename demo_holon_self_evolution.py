#!/usr/bin/env python3
"""
HolonPolis 自演化能力演示

展示 Holon 如何：
1. 请求演化新技能 (RGV 流程)
2. 自我分析并识别改进点
3. 组合现有技能形成新能力
4. 从失败中学习并自动重试
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.blueprints import EvolutionStrategy
from holonpolis.runtime.holon_runtime import HolonRuntime


async def demo_self_evolution():
    """演示 Holon 的自演化能力。"""
    print("=" * 80)
    print("🧬 HOLOPOLIS 自演化能力演示")
    print("=" * 80)

    # 创建一个具备演化能力的 Holon
    blueprint = Blueprint(
        blueprint_id="blueprint_demo_001",
        holon_id="holon_self_evolver_001",
        species_id="self_improver",
        name="Self-Evolving Assistant",
        purpose="Demonstrate self-evolution capabilities through RGV pipeline",
        boundary=Boundary(
            allowed_tools=["file_read", "file_write", "code_generate", "skill_evolve"],
            allow_file_write=True,
            max_tokens_per_episode=16000,
        ),
        evolution_policy=EvolutionPolicy(
            strategy=EvolutionStrategy.AGGRESSIVE,
            auto_promote_to_global=False,
            require_tests=True,
            max_evolution_attempts=5,
        ),
    )

    print("\n🥚 Step 1: 创建具备演化能力的 Holon")
    print(f"   Holon ID: {blueprint.holon_id}")
    print(f"   物种: {blueprint.species_id}")
    print(f"   演化策略: {blueprint.evolution_policy.strategy.value}")
    print(f"   需要测试: {blueprint.evolution_policy.require_tests}")

    # 初始化 Holon 的数据库表
    from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory
    factory = get_lancedb_factory()
    factory.init_holon_tables(blueprint.holon_id)

    holon = HolonRuntime(
        holon_id=blueprint.holon_id,
        blueprint=blueprint,
    )

    # Step 2: 请求演化新技能
    print("\n🧪 Step 2: Holon 请求演化新技能")
    print("   技能: DataTransformer")
    print("   功能: 转换各种数据格式 (JSON, CSV, XML)")

    evolution_request = await holon.request_evolution(
        skill_name="DataTransformer",
        description="Transform data between formats: JSON, CSV, XML, YAML with validation",
        requirements=[
            "Parse JSON, CSV, XML, YAML formats",
            "Validate input data structure",
            "Convert between any supported formats",
            "Handle errors gracefully",
            "Preserve data integrity during conversion",
        ],
        test_cases=[
            {
                "name": "json_to_csv",
                "input": {"data": [{"name": "John", "age": 30}]},
                "expected": "name,age\nJohn,30",
            },
            {
                "name": "csv_to_json",
                "input": "name,age\nJohn,30",
                "expected": {"data": [{"name": "John", "age": "30"}]},
            },
            {
                "name": "invalid_format",
                "input": "not valid data",
                "expected": "error",
            },
        ],
    )

    print(f"   演化请求 ID: {evolution_request.request_id}")
    print(f"   初始状态: {evolution_request.status.value}")
    print(f"   创建时间: {evolution_request.created_at}")

    # Step 3: 自我分析
    print("\n🔍 Step 3: Holon 进行自我分析")
    print("   分析最近的表现，识别需要改进的地方...")

    improvement_plan = await holon.self_improve()

    print(f"   分析状态: {improvement_plan['status']}")
    if 'metrics' in improvement_plan:
        metrics = improvement_plan['metrics']
        print(f"   总交互数: {metrics.get('total_episodes', 0)}")
        print(f"   成功率: {metrics.get('success_rate', 0):.1%}")

    if improvement_plan.get('suggestions'):
        print(f"   改进建议: {len(improvement_plan['suggestions'])} 条")
        for i, suggestion in enumerate(improvement_plan['suggestions'], 1):
            print(f"      {i}. {suggestion['type']}: {suggestion.get('reason', '')}")

    # Step 4: 组合技能
    print("\n🧩 Step 4: Holon 组合现有技能")
    print("   基于已有技能组合新能力...")

    # 假设 Holon 已经有一些技能
    await holon.remember(
        content="Skill file_reader: Read files in various formats",
        tags=["skill", "file_reader"],
        importance=0.9,
    )
    await holon.remember(
        content="Skill data_validator: Validate data structures",
        tags=["skill", "data_validator"],
        importance=0.9,
    )

    compose_request = await holon.compose_skill(
        new_skill_name="DataPipeline",
        parent_skill_ids=["file_reader", "data_validator"],
        composition_description="Read data from files, validate it, and transform to desired format in one pipeline",
    )

    print(f"   组合技能请求 ID: {compose_request.request_id}")
    print(f"   新技能: DataPipeline")
    print(f"   父技能: file_reader + data_validator")
    print(f"   状态: {compose_request.status.value}")

    # Step 5: 展示演化状态追踪
    print("\n📊 Step 5: 演化状态追踪")
    print(f"   Holon 已发起的演化请求: {len(holon.state.evolution_requests)}")
    for req_id in holon.state.evolution_requests:
        print(f"      - {req_id}")

    print(f"   已获得的技能: {len(holon.state.skills)}")
    for skill_id in holon.state.skills:
        print(f"      - {skill_id}")

    print("\n" + "=" * 80)
    print("✅ 自演化演示完成")
    print("=" * 80)
    print("\n关键点:")
    print("  1. Holon 通过 request_evolution() 发起 RGV 演化")
    print("  2. Red 阶段: 生成测试用例定义期望行为")
    print("  3. Green 阶段: 生成代码通过测试")
    print("  4. Verify 阶段: 安全扫描和验证")
    print("  5. Persist: 技能保存到本地目录")
    print("  6. self_improve() 分析表现并识别改进点")
    print("  7. compose_skill() 组合现有技能形成新能力")
    print("\n这种架构确保每个 Holon 真正具备:")
    print("  ✓ 自学习能力 (从交互中学习)")
    print("  ✓ 自改进能力 (识别并修复缺陷)")
    print("  ✓ 自组合能力 (基于已有技能构建新技能)")
    print("  ✓ 自验证能力 (RGV 确保代码质量)")

    return True


if __name__ == "__main__":
    success = asyncio.run(demo_self_evolution())
    sys.exit(0 if success else 1)
