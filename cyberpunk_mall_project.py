#!/usr/bin/env python3
"""
HolonPolis 压测 - 赛博朋克风格大型购物网站前端代码生成
使用增强的 React 项目生成器

这是对 HolonPolis 系统的大型项目生成能力的压力测试。
目标：在 C:/Temp/ 生成一个完整的赛博朋克风格购物网站
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.services.evolution_service import EvolutionService

# 压测配置
OUTPUT_DIR = Path("C:/Temp/cyberpunk-mall")
PROJECT_NAME = "CyberPunk Mall"

# 赛博朋克主题配置
CYBERPUNK_THEME = {
    "colors": {
        "background": "#0a0a0f",
        "surface": "#12121a",
        "surfaceLight": "#1a1a25",
        "primary": "#00f0ff",      # Cyan neon
        "secondary": "#ff00a0",    # Pink neon
        "tertiary": "#a020f0",     # Purple neon
        "success": "#00ff88",
        "warning": "#ffaa00",
        "error": "#ff3333",
        "text": "#e0e0e0",
        "textMuted": "#888888",
        "border": "#2a2a3a",
    },
    "fonts": {
        "heading": "'Orbitron', 'Share Tech Mono', monospace",
        "body": "'Rajdhani', 'Inter', system-ui, sans-serif",
        "mono": "'Fira Code', 'JetBrains Mono', monospace",
    },
    "effects": {
        "glowCyan": "0 0 10px #00f0ff, 0 0 20px #00f0ff40",
        "glowPink": "0 0 10px #ff00a0, 0 0 20px #ff00a040",
        "glowPurple": "0 0 10px #a020f0, 0 0 20px #a020f040",
    }
}

# 大型项目需求定义
PROJECT_REQUIREMENTS = [
    # ===== 技术栈 =====
    "React 18 + TypeScript + Vite",
    "React Router DOM for SPA navigation",
    "Tailwind CSS for styling",
    "Lucide React for icons",

    # ===== 核心页面 (6个页面) =====
    "Home page with animated hero banner, featured products grid, category showcase, promotional banners",
    "Products catalog page with advanced filters (category, price range, rating), sorting, pagination, search",
    "Product detail page with image gallery, specifications table, reviews, related products, add to cart",
    "Shopping cart page with editable quantities, item removal, price calculations, checkout button",
    "Checkout page with multi-step form (shipping, payment, review), validation, order summary",
    "User auth pages (Login/Register) with form validation, password strength, error handling",

    # ===== 共享组件 (15+ 组件) =====
    "Header component with animated logo, navigation menu, search bar, cart icon with badge, user menu",
    "Footer component with newsletter signup, site links, social icons, payment methods",
    "ProductCard component with hover effects, quick add button, price display, rating stars",
    "NeonButton component with cyan/pink/purple variants, glow animation, loading state",
    "CartDrawer slide-out panel with item list, quantity controls, total price, checkout CTA",
    "HeroBanner with glitch text effect, animated background, call-to-action buttons",
    "SearchBar with autocomplete suggestions, search history, voice search icon",
    "FilterSidebar with collapsible sections, price range slider, checkbox filters",
    "StarRating component with half-star support, review count display",
    "ImageGallery with zoom, thumbnail navigation, fullscreen view",
    "Toast notification system for cart additions, errors, success messages",
    "Loading skeleton screens for products and pages",
    "Breadcrumb navigation for deep linking",
    "Pagination component with page numbers, prev/next, ellipsis",
    "Modal dialog for quick product view, confirmations",

    # ===== 赛博朋克风格要求 =====
    "Dark background (#0a0a0f) with grid pattern overlay",
    "Neon cyan (#00f0ff) primary color with glow effects",
    "Neon pink (#ff00a0) secondary color for accents",
    "Neon purple (#a020f0) tertiary color for highlights",
    "Glitch text effects on headings and important text",
    "Animated scanline overlay for retro CRT feel",
    "Glow pulse animations on interactive elements",
    "Cyberpunk fonts: Orbitron for headings, Rajdhani for body",
    "Tech-pattern borders with gradient edges",
    "Holographic card effects with shimmer",

    # ===== 功能特性 =====
    "Add to cart with animation feedback",
    "Remove from cart with confirmation",
    "Update quantity with +/- buttons",
    "Real-time price calculations with discounts",
    "Form validation with error messages",
    "Responsive design (mobile, tablet, desktop breakpoints)",
    "Local storage persistence for cart",
    "Keyboard navigation support",
    "Loading states and error boundaries",
]


async def run_stress_test():
    """执行压测 - 生成大型 React 项目。"""
    print("=" * 80)
    print("🧪 HOLOPOLIS 压测 - 赛博朋克购物网站生成")
    print("=" * 80)
    print(f"项目: {PROJECT_NAME}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"需求项: {len(PROJECT_REQUIREMENTS)}")
    print("-" * 80)

    start_time = time.time()

    # 清理之前的输出
    if OUTPUT_DIR.exists():
        import shutil
        print("🧹 清理旧项目...")
        shutil.rmtree(OUTPUT_DIR)

    # 创建进化服务
    service = EvolutionService()

    # 执行项目演化
    print("\n🚀 启动项目生成...")
    print("⏳ 这将生成完整的 React 项目 (约 20+ 文件)")
    print()

    result = await service.evolve_react_project_auto(
        project_name=PROJECT_NAME,
        description="""
A large-scale cyberpunk-themed e-commerce shopping website.
Features: product catalog, shopping cart, checkout flow, user authentication.
Style: Cyberpunk 2077 inspired with neon cyan/pink/purple colors, dark theme, grid layouts, tech aesthetics.
        """.strip(),
        requirements=PROJECT_REQUIREMENTS,
        target_dir=OUTPUT_DIR,
        provider_id="ollama-local",
        style_theme=CYBERPUNK_THEME,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 80)

    if result.success:
        print("✅ 压测成功 - 项目生成完成!")
        print("=" * 80)

        # 统计生成结果
        file_stats = {"code": 0, "config": 0, "style": 0, "total": 0}
        total_lines = 0

        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                file_stats["total"] += 1
                if f.suffix in ['.ts', '.tsx', '.js', '.jsx']:
                    file_stats["code"] += 1
                    total_lines += len(f.read_text(encoding="utf-8").splitlines())
                elif f.suffix in ['.css', '.scss']:
                    file_stats["style"] += 1
                    total_lines += len(f.read_text(encoding="utf-8").splitlines())
                elif f.suffix in ['.json', '.js']:
                    file_stats["config"] += 1

        print(f"\n📊 生成统计:")
        print(f"   耗时: {elapsed:.1f} 秒")
        print(f"   总文件: {file_stats['total']}")
        print(f"   代码文件: {file_stats['code']}")
        print(f"   样式文件: {file_stats['style']}")
        print(f"   配置文件: {file_stats['config']}")
        print(f"   代码行数: {total_lines}")

        print(f"\n📁 项目结构:")
        for item in sorted(OUTPUT_DIR.rglob("*")):
            if item.is_file():
                rel = item.relative_to(OUTPUT_DIR)
                depth = len(rel.parts) - 1
                indent = "  " * depth
                print(f"   {indent}{rel.name}")

        print(f"\n🚀 启动命令:")
        print(f"   cd {OUTPUT_DIR}")
        print(f"   npm install")
        print(f"   npm run dev")
        print(f"\n🌐 访问: http://localhost:5173")

        print(f"\n✨ 赛博朋克主题:")
        print(f"   背景色: {CYBERPUNK_THEME['colors']['background']}")
        print(f"   主色:   {CYBERPUNK_THEME['colors']['primary']} (Cyan)")
        print(f"   辅色:   {CYBERPUNK_THEME['colors']['secondary']} (Pink)")
        print(f"   强调色: {CYBERPUNK_THEME['colors']['tertiary']} (Purple)")

        # 验证关键文件
        key_files = [
            "package.json",
            "src/main.tsx",
            "src/App.tsx",
            "src/index.css",
            "tailwind.config.js",
            "src/pages/Home.tsx",
            "src/components/Header.tsx",
        ]

        print(f"\n✅ 关键文件检查:")
        all_exist = True
        for key_file in key_files:
            exists = (OUTPUT_DIR / key_file).exists()
            status = "✓" if exists else "✗"
            print(f"   {status} {key_file}")
            all_exist = all_exist and exists

        if all_exist:
            print(f"\n🎉 所有关键文件已生成!")
            return True
        else:
            print(f"\n⚠️ 部分文件缺失")
            return False

    else:
        print("❌ 压测失败 - 项目生成出错")
        print("=" * 80)
        print(f"错误阶段: {result.phase}")
        print(f"错误信息: {result.error_message}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_stress_test())
    sys.exit(0 if success else 1)
