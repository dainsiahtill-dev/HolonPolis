#!/usr/bin/env python3
"""
完整流程：
1. Holon 学习前端 UI 项目（全量）
2. 生成 Dashboard 网站（应用学习的组件）
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.services.repository_learner import RepositoryLearningService
from holonpolis.services.evolution_service import EvolutionService
from holonpolis.services.memory_service import MemoryService
from holonpolis.services.holon_service import HolonService

# 配置
UI_PROJECT_PATH = r"C:\Users\dains\Downloads\Minimal_JavaScript_v7.6.1-ksecbk\Minimal_JavaScript_v7.6.1\vite-js"
HOLON_ID = "holon_dashboard_builder_001"
OUTPUT_DIR = Path("C:/Temp/HolonProjects/dashboard-site")


async def step1_learn_ui_project():
    """步骤 1: Holon 学习 UI 项目。"""
    print("="*70)
    print("📚 步骤 1: Holon 全量学习 UI 项目")
    print("="*70)
    print(f"目标: {UI_PROJECT_PATH}")
    print()

    # 确保 Holon 存在
    holon_svc = HolonService()
    if not holon_svc.holon_exists(HOLON_ID):
        from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
        from holonpolis.domain.blueprints import EvolutionStrategy

        blueprint = Blueprint(
            blueprint_id='blueprint_dashboard_builder',
            holon_id=HOLON_ID,
            species_id='frontend_developer',
            name='Dashboard Builder',
            purpose='Build dashboard websites using learned UI components',
            boundary=Boundary(
                allowed_tools=['file_read', 'file_write', 'code_generate'],
                denied_tools=[],
                max_episodes_per_hour=200,
                max_tokens_per_episode=100000,
                allow_file_write=True,
                allow_network=False,
                allow_subprocess=False,
            ),
            evolution_policy=EvolutionPolicy(
                strategy=EvolutionStrategy.BALANCED,
                auto_promote_to_global=False,
                require_tests=True,
                max_evolution_attempts=5,
            ),
            initial_memory_tags=['dashboard', 'ui-components', 'react'],
        )
        await holon_svc.create_holon(blueprint)
        print(f"✅ Holon 创建: {HOLON_ID}")
    else:
        print(f"✅ Holon 已存在: {HOLON_ID}")

    # 全量学习 UI 项目
    service = RepositoryLearningService()

    result = await service.learn(
        holon_id=HOLON_ID,
        repo_url=UI_PROJECT_PATH,
        branch="main",
        depth=5,  # 最深层级
        focus_areas=[
            "components",
            "ui-patterns",
            "layout",
            "styling",
            "animation",
            "dashboard-elements",
            "navigation",
            "forms",
            "data-display",
        ],
    )

    if result.success:
        print(f"\n✅ UI 项目学习完成!")
        print(f"   组件数: {len(result.analysis.key_patterns)}")
        print(f"   技术栈: {result.analysis.languages}")
        print(f"   存储记忆: {result.memories_created} 条")

        # 显示学习到的关键内容
        memory = MemoryService(HOLON_ID)
        learnings = await memory.recall("UI components", top_k=5)
        print(f"\n🧠 学习要点预览:")
        for i, mem in enumerate(learnings[:3], 1):
            content = mem.get('content', '')[:60]
            print(f"   {i}. {content}...")

        return True
    else:
        print(f"\n❌ 学习失败: {result.error_message}")
        return False


async def step2_generate_dashboard():
    """步骤 2: 生成 Dashboard 网站。"""
    print("\n" + "="*70)
    print("🎨 步骤 2: 生成 Dashboard 网站")
    print("="*70)
    print(f"输出: {OUTPUT_DIR}")
    print()

    # 检索学习到的 UI 知识
    memory = MemoryService(HOLON_ID)
    ui_knowledge = await memory.recall("components patterns", top_k=10)

    # 构建知识上下文
    knowledge_context = "\n".join([
        f"- {mem.get('content', '')[:100]}"
        for mem in ui_knowledge[:5]
    ])

    service = EvolutionService()

    # 清理之前的结果
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    # 生成 Dashboard 项目
    result = await service.evolve_typescript_project_auto(
        project_name="Admin Dashboard",
        description="Complete admin dashboard with sidebar navigation, data cards, charts, tables, and modern UI",
        requirements=[
            "React + Vite + TypeScript setup",
            "Responsive sidebar navigation with icons",
            "Dashboard overview page with stats cards",
            "Data table with sorting and pagination",
            "Chart components (line chart, pie chart)",
            "User profile section",
            "Settings panel",
            "Dark/Light theme support",
            "Responsive design (mobile, tablet, desktop)",
            "Modern CSS with animations",
            "Component-based architecture",
            "Use proper TypeScript interfaces",
        ],
        target_dir=OUTPUT_DIR,
        provider_id="ollama-local",
    )

    if result.success:
        print(f"✅ Dashboard 生成成功!")
        print(f"   位置: {OUTPUT_DIR}")
        print(f"   代码: {result.code_path}")

        # 显示生成的文件结构
        print(f"\n📁 生成的文件结构:")
        for item in sorted(OUTPUT_DIR.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(OUTPUT_DIR)
                print(f"   {rel_path}")

        # 读取生成的代码预览
        code_file = OUTPUT_DIR / "src" / "index.ts"
        if code_file.exists():
            code = code_file.read_text()
            lines = len(code.splitlines())
            print(f"\n📊 代码统计:")
            print(f"   总行数: {lines}")
            print(f"   文件大小: {len(code)} bytes")

        return True
    else:
        print(f"❌ 生成失败: {result.error_message}")
        return False


async def step3_verify_integration():
    """步骤 3: 验证学习到的知识是否被应用。"""
    print("\n" + "="*70)
    print("✅ 步骤 3: 验证集成")
    print("="*70)

    # 检查生成的代码是否包含学习到的模式
    code_file = OUTPUT_DIR / "src" / "index.ts"
    if not code_file.exists():
        print("❌ 未找到生成的代码文件")
        return False

    code = code_file.read_text()

    # 检查关键元素
    checks = {
        "React imports": "import" in code and "react" in code.lower(),
        "Components": "component" in code.lower() or "function" in code,
        "TypeScript types": "interface" in code or "type " in code,
        "Dashboard elements": "dashboard" in code.lower() or "admin" in code.lower(),
        "Responsive design": "media" in code or "responsive" in code.lower() or "grid" in code,
        "Modern syntax": "=>" in code,  # Arrow functions
    }

    print("\n🔍 代码质量检查:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False

    # 检索 Holon 记忆，看它是否记得学过的组件
    memory = MemoryService(HOLON_ID)
    recall_results = await memory.recall("dashboard", top_k=3)
    print(f"\n💾 Holon 记忆检索:")
    print(f"   找到 {len(recall_results)} 条 dashboard 相关记忆")

    return all_passed


async def main():
    """执行完整流程。"""
    print("\n" + "🚀"*35)
    print("HOLON DASHBOARD 项目")
    print("流程: 学习 UI → 生成 Dashboard → 验证")
    print("🚀"*35 + "\n")

    # 步骤 1: 学习
    step1_ok = await step1_learn_ui_project()
    if not step1_ok:
        print("\n❌ 步骤 1 失败，终止")
        return False

    # 步骤 2: 生成
    step2_ok = await step2_generate_dashboard()
    if not step2_ok:
        print("\n❌ 步骤 2 失败，终止")
        return False

    # 步骤 3: 验证
    step3_ok = await step3_verify_integration()

    # 最终报告
    print("\n" + "="*70)
    print("📊 最终报告")
    print("="*70)
    print(f"步骤 1 (学习 UI): {'✅ 通过' if step1_ok else '❌ 失败'}")
    print(f"步骤 2 (生成 Dashboard): {'✅ 通过' if step2_ok else '❌ 失败'}")
    print(f"步骤 3 (验证集成): {'✅ 通过' if step3_ok else '❌ 失败'}")

    if step1_ok and step2_ok:
        print(f"\n🎉 项目完成!")
        print(f"   Dashboard 位置: {OUTPUT_DIR}")
        print(f"   Holon ID: {HOLON_ID}")
        print(f"\n启动命令:")
        print(f"   cd {OUTPUT_DIR}")
        print(f"   npm install")
        print(f"   npm run dev")

    return step1_ok and step2_ok and step3_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
