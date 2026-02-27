#!/usr/bin/env python3
"""
Holon 代码仓库学习工具

用法:
    python learn_repo.py <holon_id> <repo_url> [options]

示例:
    python learn_repo.py holon_001 https://github.com/vuejs/core --focus architecture,patterns
    python learn_repo.py holon_001 https://github.com/expressjs/express --depth 5
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.services.repository_learner import RepositoryLearningService
from holonpolis.services.holon_service import HolonService


async def main():
    parser = argparse.ArgumentParser(
        description="让 Holon 学习指定的代码仓库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 学习 Vue.js 核心库
  python learn_repo.py holon_001 https://github.com/vuejs/core

  # 深度学习，关注架构模式
  python learn_repo.py holon_001 https://github.com/expressjs/express --depth 5 --focus architecture,testing

  # 学习特定分支
  python learn_repo.py holon_001 https://github.com/facebook/react --branch main
        """
    )

    parser.add_argument("holon_id", nargs="?", help="Holon ID (例如: holon_001)")
    parser.add_argument("repo_url", nargs="?", help="代码仓库 URL (例如: https://github.com/...)")
    parser.add_argument("--branch", default="main", help="分支名 (默认: main)")
    parser.add_argument("--depth", type=int, default=3, help="分析深度 1-5 (默认: 3)")
    parser.add_argument("--focus", type=str, default="", help="关注领域，逗号分隔 (例如: architecture,patterns,testing)")
    parser.add_argument("--list-holons", action="store_true", help="列出所有可用的 Holons")

    args = parser.parse_args()

    # 列出 Holons
    if args.list_holons:
        service = HolonService()
        holons = service.list_holons()
        print("\n📋 可用的 Holons:")
        print("-" * 60)
        for h in holons:
            print(f"  {h['holon_id']}: {h.get('name', 'Unnamed')}")
            print(f"    物种: {h.get('species_id', 'unknown')}")
            print(f"    目的: {h.get('purpose', 'N/A')[:50]}...")
            print()
        return

    # 验证参数
    if not args.holon_id or not args.repo_url:
        parser.print_help()
        print("\n❌ 错误: 必须提供 holon_id 和 repo_url")
        sys.exit(1)

    # 验证 Holon 存在
    holon_service = HolonService()
    if not holon_service.holon_exists(args.holon_id):
        print(f"❌ 错误: Holon '{args.holon_id}' 不存在")
        print("使用 --list-holons 查看可用的 Holons")
        sys.exit(1)

    # 解析关注领域
    focus_areas = None
    if args.focus:
        focus_areas = [f.strip() for f in args.focus.split(",")]

    print("="*70)
    print("📚 Holon 代码仓库学习")
    print("="*70)
    print(f"Holon ID: {args.holon_id}")
    print(f"仓库: {args.repo_url}")
    print(f"分支: {args.branch}")
    print(f"深度: {args.depth}")
    if focus_areas:
        print(f"关注: {', '.join(focus_areas)}")
    print("="*70)
    print()

    # 开始学习
    service = RepositoryLearningService()

    try:
        result = await service.learn(
            holon_id=args.holon_id,
            repo_url=args.repo_url,
            branch=args.branch,
            depth=args.depth,
            focus_areas=focus_areas,
        )

        if result.success:
            print(f"✅ 学习成功!\n")
            print(f"📊 仓库信息:")
            print(f"   名称: {result.analysis.repo_name}")
            print(f"   文件数: {result.analysis.total_files}")
            print(f"   代码行数: {result.analysis.total_lines}")
            print()

            print(f"💻 技术栈:")
            for lang, count in sorted(result.analysis.languages.items(), key=lambda x: -x[1])[:5]:
                print(f"   - {lang}: {count} 文件")
            print()

            print(f"🏗️ 架构: {result.analysis.architecture}")
            print()

            if result.analysis.key_patterns:
                print(f"🔍 识别模式: {', '.join(result.analysis.key_patterns)}")
                print()

            if result.analysis.learnings:
                print(f"🧠 学习要点:")
                for i, learning in enumerate(result.analysis.learnings, 1):
                    print(f"   {i}. {learning}")
                print()

            print(f"💾 已存储到 Holon 记忆: {result.memories_created} 条")

        else:
            print(f"❌ 学习失败: {result.error_message}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户取消")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
