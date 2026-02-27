#!/usr/bin/env python3
"""Holon 构建赛博朋克购物网站"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.services.evolution_service import EvolutionService

HOLON_ID = "holon_deep_learner_001"
OUTPUT_DIR = Path("C:/Temp/HolonProjects/cyberpunk-mall")


async def main():
    print("="*70)
    print("🛍️ Holon 构建赛博朋克购物网站")
    print("="*70)

    service = EvolutionService()

    result = await service.evolve_react_project(
        project_name="CyberPunk Mall",
        description="赛博朋克风格大型购物网站，具备完整功能",
        requirements=[
            "首页轮播图和推荐商品",
            "商品分类浏览",
            "商品详情页",
            "购物车功能",
            "用户登录注册",
            "赛博朋克风格：霓虹灯效果、暗色背景、科技感",
            "响应式设计",
            "React hooks 状态管理",
        ],
        target_dir=OUTPUT_DIR,
        provider_id="ollama-local",
    )

    if result.success:
        print("\n✅ 生成成功!")
        print(f"位置: {OUTPUT_DIR}")

        # 统计代码
        lines = 0
        for f in OUTPUT_DIR.rglob("*.tsx"):
            if f.is_file():
                lines += len(f.read_text().splitlines())
        print(f"代码行数: {lines}")

        print("\n启动命令:")
        print(f"  cd {OUTPUT_DIR}")
        print("  npm install")
        print("  npm run dev")
    else:
        print(f"\n❌ 失败: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
