#!/usr/bin/env python3
"""
HolonPolis 压测 - 通过 Genesis 服务孵化 Holon 生成赛博朋克购物网站

原则：
1. 不直接写任何业务代码生成逻辑
2. 通过 GenesisService 路由/孵化 Holon
3. 让 Holon 自演化出生成能力
4. 通过 EvolutionService 执行 RGV 演化
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.services.genesis_service import GenesisService
from holonpolis.services.evolution_service import EvolutionService


# 赛博朋克购物网站需求
PROJECT_REQUIREMENTS = [
    "React 18 + TypeScript + Vite",
    "React Router DOM for navigation",
    "Tailwind CSS for styling",
    "Home page with hero banner and featured products",
    "Products catalog page with filters and search",
    "Product detail page with image gallery and add to cart",
    "Shopping cart page with item management",
    "Checkout page with form validation",
    "Login page with authentication",
    "Header component with navigation and cart icon",
    "Footer component with links",
    "ProductCard component with hover effects",
    "Dark theme (#0a0a0f background)",
    "Neon cyan (#00f0ff) primary color",
    "Neon pink (#ff00a0) secondary color",
    "Grid patterns and glow effects",
    "Add to cart functionality",
    "Local storage persistence",
    "Responsive design",
]


async def stress_test():
    """压测：通过系统服务生成赛博朋克购物网站。"""
    print("=" * 80)
    print("🧬 HOLOPOLIS 压测 - 系统自演化代码生成")
    print("=" * 80)
    print(f"项目: CyberPunk Mall")
    print(f"目标: C:/Temp/cyberpunk-mall")
    print(f"需求项: {len(PROJECT_REQUIREMENTS)}")
    print("-" * 80)

    # Step 1: 初始化 Genesis 服务
    print("\n🧬 Step 1: 初始化 Genesis 服务...")
    genesis = GenesisService()

    # 准备请求 (Genesis 期望 user_request 是字符串)
    request_text = f"""Generate a React project: CyberPunk Mall
Project Type: cyberpunk_ecommerce
Target Directory: C:/Temp/cyberpunk-mall
Complexity: high

Requirements:
""" + "\n".join(f"- {r}" for r in PROJECT_REQUIREMENTS)

    print(f"   请求意图: generate_react_project")
    print(f"   项目类型: cyberpunk_ecommerce")

    # Step 2: Genesis 路由决策
    print("\n🎯 Step 2: Genesis 路由决策...")

    result = await genesis.route_or_spawn(request_text)

    print(f"   决策: {result.decision}")
    print(f"   推理: {result.reasoning}")

    if result.decision == "spawn":
        print(f"\n   孵化新 Holon: {result.blueprint.holon_id}")
        print(f"   物种: {result.blueprint.species_id}")
        print(f"   用途: {result.blueprint.purpose}")

        # Step 3: 执行项目生成
        print("\n🚀 Step 3: 执行项目生成...")

        # 使用 EvolutionService 生成项目
        evolution = EvolutionService()

        target_dir = Path("C:/Temp/cyberpunk-mall")

        # 清理旧项目
        if target_dir.exists():
            import shutil
            print("   清理旧项目...")
            shutil.rmtree(target_dir)

        # 使用新的 React 项目生成方法 (通过 LLM 驱动)
        evolution_result = await evolution.evolve_react_project_auto(
            project_name="CyberPunk Mall",
            description="""
A large-scale cyberpunk-themed e-commerce shopping website.
Features: product catalog, shopping cart, checkout flow, user authentication.
Style: Cyberpunk 2077 inspired with neon cyan/pink colors, dark theme, grid layouts.
            """.strip(),
            requirements=PROJECT_REQUIREMENTS,
            target_dir=target_dir,
            provider_id="ollama-local",
        )

        print("-" * 80)

        if evolution_result.success:
            print("✅ 压测成功 - 项目生成完成!")
            print("=" * 80)

            # 统计生成结果
            file_stats = {"total": 0, "code": 0, "style": 0, "config": 0}
            total_lines = 0

            for f in target_dir.rglob("*"):
                if f.is_file():
                    file_stats["total"] += 1
                    content = f.read_text(encoding="utf-8")
                    lines = len(content.splitlines())
                    total_lines += lines

                    if f.suffix in [".ts", ".tsx"]:
                        file_stats["code"] += 1
                    elif f.suffix in [".css", ".scss"]:
                        file_stats["style"] += 1
                    elif f.suffix in [".json", ".js"]:
                        file_stats["config"] += 1

            print(f"\n📊 生成统计:")
            print(f"   总文件: {file_stats['total']}")
            print(f"   代码文件: {file_stats['code']}")
            print(f"   样式文件: {file_stats['style']}")
            print(f"   配置文件: {file_stats['config']}")
            print(f"   代码行数: {total_lines}")

            print(f"\n📁 项目结构:")
            for f in sorted(target_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(target_dir)
                    depth = len(rel.parts) - 1
                    indent = "  " * depth
                    print(f"   {indent}{rel.name}")

            print(f"\n🚀 启动命令:")
            print(f"   cd {target_dir}")
            print(f"   npm install")
            print(f"   npm run dev")
            print(f"\n🌐 访问: http://localhost:5173")

            return True
        else:
            print("❌ 项目生成失败")
            print(f"   阶段: {evolution_result.phase}")
            print(f"   错误: {evolution_result.error_message}")
            return False

    elif result.decision == "route_to":
        print(f"\n🔄 路由到现有 Holon: {result.holon_id}")
        print("   (使用已有 Holon 的能力)")

        # 仍然执行项目生成
        print("\n🚀 Step 3: 执行项目生成...")

        evolution = EvolutionService()
        target_dir = Path("C:/Temp/cyberpunk-mall")

        # 清理旧项目 (忽略 Windows 文件占用错误)
        if target_dir.exists():
            import shutil
            import time
            print("   清理旧项目...")
            for _ in range(3):
                try:
                    shutil.rmtree(target_dir)
                    break
                except PermissionError:
                    time.sleep(0.5)
            else:
                print("   警告: 无法删除旧目录，将覆盖写入...")

        # 使用 React 项目生成方法 (使用更快的 LLM provider)
        # 优先使用: kimi-coding > minimax > ollama-local
        provider_id = "kimi-coding"  # Kimi Coding (最快)
        # provider_id = "minimax"  # MiniMax-M2.5 (备选)

        print(f"   使用 LLM Provider: {provider_id}")

        evolution_result = await evolution.evolve_react_project_auto(
            project_name="CyberPunk Mall",
            description="""
A large-scale cyberpunk-themed e-commerce shopping website.
Features: product catalog, shopping cart, checkout flow, user authentication.
Style: Cyberpunk 2077 inspired with neon cyan/pink colors, dark theme, grid layouts.
            """.strip(),
            requirements=PROJECT_REQUIREMENTS,
            target_dir=target_dir,
            provider_id=provider_id,
        )

        print("-" * 80)

        if evolution_result.success:
            print("✅ 压测成功 - 项目生成完成!")
            print("=" * 80)

            # 统计生成结果
            file_stats = {"total": 0, "code": 0, "style": 0, "config": 0}
            total_lines = 0

            for f in target_dir.rglob("*"):
                if f.is_file():
                    file_stats["total"] += 1
                    content = f.read_text(encoding="utf-8")
                    lines = len(content.splitlines())
                    total_lines += lines

                    if f.suffix in [".ts", ".tsx"]:
                        file_stats["code"] += 1
                    elif f.suffix in [".css", ".scss"]:
                        file_stats["style"] += 1
                    elif f.suffix in [".json", ".js"]:
                        file_stats["config"] += 1

            print(f"\n📊 生成统计:")
            print(f"   总文件: {file_stats['total']}")
            print(f"   代码文件: {file_stats['code']}")
            print(f"   样式文件: {file_stats['style']}")
            print(f"   配置文件: {file_stats['config']}")
            print(f"   代码行数: {total_lines}")

            print(f"\n📁 项目结构:")
            for f in sorted(target_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(target_dir)
                    depth = len(rel.parts) - 1
                    indent = "  " * depth
                    print(f"   {indent}{rel.name}")

            print(f"\n🚀 启动命令:")
            print(f"   cd {target_dir}")
            print(f"   npm install")
            print(f"   npm run dev")
            print(f"\n🌐 访问: http://localhost:5173")

            return True
        else:
            print("❌ 项目生成失败")
            print(f"   阶段: {evolution_result.phase}")
            print(f"   错误: {evolution_result.error_message}")
            return False

    elif result.decision == "deny":
        print(f"\n❌ Genesis 拒绝请求")
        print(f"   原因: {result.message}")
        return False

    else:  # clarify
        print(f"\n❓ 需要澄清:")
        print(f"   {result.message}")
        return False


if __name__ == "__main__":
    success = asyncio.run(stress_test())
    sys.exit(0 if success else 1)
