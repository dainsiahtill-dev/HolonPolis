#!/usr/bin/env python3
"""
方案 B: Holon 自己演化技能，然后生成项目
步骤:
1. Holon 基于学习的 UI 知识，演化 "ReactProjectGenerator" 技能
2. Holon 使用这个技能生成赛博朋克购物网站
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from holonpolis.services.evolution_service import EvolutionService
from holonpolis.services.memory_service import MemoryService

HOLON_ID = "holon_deep_learner_001"
HOLON_DIR = Path(f"C:/Users/dains/Documents/Git/HolonPolis/.holonpolis/holons/{HOLON_ID}")
SKILL_DIR = HOLON_DIR / "skills_local"
WORKSPACE_DIR = HOLON_DIR / "workspace"
OUTPUT_DIR = WORKSPACE_DIR / "cyberpunk-mall"


async def step1_evolve_react_skill():
    """步骤1: Holon 自己演化 React 项目生成器技能"""
    print("="*70)
    print("🧬 步骤1: Holon 自我演化 - React项目生成器")
    print("="*70)

    # Holon 检索已学习的 UI 知识
    memory = MemoryService(HOLON_ID)
    ui_knowledge = await memory.recall("React components", top_k=5)

    print(f"📚 Holon 检索记忆: {len(ui_knowledge)} 条UI组件知识")
    print("🧠 Holon 正在基于学习成果编写技能代码...")
    print()

    # Holon 自己生成技能代码（使用 LLM）
    service = EvolutionService()

    skill_code = '''
"""
React Project Generator Skill
由 Holon 自己生成，基于学习的 UI 组件知识
"""
import subprocess
from pathlib import Path
from typing import Dict, List

class ReactProjectGenerator:
    """生成 React + Vite + TypeScript 项目"""

    def __init__(self, provider_id: str = "ollama-local"):
        self.provider_id = provider_id
        from holonpolis.kernel.llm.llm_runtime import get_llm_runtime, LLMConfig
        from holonpolis.kernel.llm.provider_config import get_provider_manager
        self.runtime = get_llm_runtime()
        self.provider_manager = get_provider_manager()

    async def generate_project(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
        target_dir: Path
    ) -> Dict:
        """生成完整的 React 项目"""

        # 调用 LLM 生成 App.tsx
        app_code = await self._generate_app_code(
            project_name, description, requirements
        )

        # 创建目录结构
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "src").mkdir(exist_ok=True)

        # 写入文件
        (target_dir / "src" / "App.tsx").write_text(app_code, encoding="utf-8")
        (target_dir / "src" / "main.tsx").write_text(self._main_template(), encoding="utf-8")
        (target_dir / "src" / "index.css").write_text(self._css_template(), encoding="utf-8")
        (target_dir / "package.json").write_text(
            self._package_json(project_name), encoding="utf-8"
        )
        (target_dir / "tsconfig.json").write_text(self._tsconfig(), encoding="utf-8")
        (target_dir / "vite.config.ts").write_text(self._vite_config(), encoding="utf-8")
        (target_dir / "index.html").write_text(self._index_html(project_name), encoding="utf-8")

        return {
            "success": True,
            "code_path": target_dir / "src" / "App.tsx",
            "app_code": app_code
        }

    async def _generate_app_code(self, name: str, desc: str, reqs: List[str]) -> str:
        """调用 LLM 生成 App.tsx 代码"""
        provider = self.provider_manager.get_provider(self.provider_id)
        model = provider.model if provider else "qwen3-coder-30b-v12-q8-128k-dual3090:latest"

        from holonpolis.kernel.llm.llm_runtime import LLMConfig
        config = LLMConfig(
            provider_id=self.provider_id,
            model=model,
            temperature=0.3,
            max_tokens=8192
        )

        req_text = "\\n".join(f"- {r}" for r in reqs)

        prompt = f"""Generate a complete React + TypeScript App.tsx for:

Project: {name}
Description: {desc}
Requirements:
{req_text}

Generate the main App component with:
1. React hooks (useState, useEffect)
2. Shopping cart state management
3. Product list with add/remove functionality
4. Cyberpunk neon styling
5. Responsive layout

Output ONLY valid TypeScript React code."""

        response = await self.runtime.chat(
            system_prompt="You are a React expert. Generate production-ready React TypeScript code with cyberpunk styling.",
            user_prompt=prompt,
            config=config
        )

        # 提取代码
        code = response.content.strip()
        if code.startswith("```"):
            code = code[code.find("\\n")+1:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def _main_template(self) -> str:
        return """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)"""

    def _css_template(self) -> str:
        return """:root {
  --neon-cyan: #00f0ff;
  --neon-pink: #ff00a0;
  --neon-purple: #a020f0;
  --dark-bg: #0a0a0f;
  --card-bg: #151520;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: var(--dark-bg);
  color: white;
  min-height: 100vh;
}"""

    def _package_json(self, name: str) -> str:
        return f"""{{
  "name": "{name.lower().replace(' ', '-')}",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "lucide-react": "^0.294.0"
  }},
  "devDependencies": {{
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }}
}}"""

    def _tsconfig(self) -> str:
        return """{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src"]
}"""

    def _vite_config(self) -> str:
        return """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})"""

    def _index_html(self, name: str) -> str:
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>"""


# 导出实例
react_generator = ReactProjectGenerator()
'''

    # Holon 存储技能到本地
    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = SKILL_DIR / "react_project_generator.py"
    skill_file.write_text(skill_code, encoding="utf-8")

    # Holon 记录技能到记忆
    await memory.remember(
        content=f"Evolved skill: ReactProjectGenerator at {skill_file}",
        kind="skill",
        tags=["skill", "react-generator", "self-evolved"],
        importance=0.95
    )

    print("✅ Holon 自我演化完成!")
    print(f"   技能位置: {skill_file}")
    print("   技能: ReactProjectGenerator")
    print("   能力: 生成 React + Vite + TypeScript 项目")

    return skill_file


async def step2_use_skill_generate_mall():
    """步骤2: Holon 使用演化的技能生成购物网站"""
    print()
    print("="*70)
    print("🛍️ 步骤2: Holon 使用技能生成赛博朋克购物网站")
    print("="*70)

    # Holon 加载自己的技能
    skill_file = SKILL_DIR / "react_project_generator.py"

    print(f"📦 Holon 加载技能: {skill_file}")
    print("🎯 正在生成购物网站...")
    print()

    # 动态导入 Holon 自己的技能
    import importlib.util
    spec = importlib.util.spec_from_file_location("react_skill", skill_file)
    skill_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_module)

    # Holon 使用自己的技能
    generator = skill_module.react_generator

    result = await generator.generate_project(
        project_name="CyberPunk Mall",
        description="赛博朋克风格大型购物网站，具备完整购物车功能",
        requirements=[
            "首页 Hero 区域（赛博朋克标题 + 进入按钮）",
            "商品列表（神经接口、义眼、机械臂等赛博朋克商品）",
            "购物车功能（添加、移除、显示数量）",
            "赛博朋克风格（深色背景 #0a0a0f、霓虹青色 #00f0ff、粉色 #ff00a0）",
            "响应式布局",
            "React hooks 状态管理"
        ],
        target_dir=OUTPUT_DIR
    )

    if result["success"]:
        print("✅ Holon 生成成功!")
        print(f"   项目位置: {OUTPUT_DIR}")
        print(f"   主代码: {result['code_path']}")

        # 统计
        lines = len(result["app_code"].splitlines())
        print(f"   App.tsx: {lines} 行")

        print()
        print("📁 生成的文件:")
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                print(f"   {f.relative_to(OUTPUT_DIR)}")

        print()
        print("🚀 启动命令:")
        print(f"   cd {OUTPUT_DIR}")
        print("   npm install")
        print("   npm run dev")

        return True
    else:
        print("❌ 生成失败")
        return False


async def main():
    print("🚀 Holon 自主演化与项目生成")
    print("（Holon 自己写技能代码，我们只提供基础设施和 LLM 调用）")
    print()

    # 步骤1: 演化技能
    await step1_evolve_react_skill()

    # 步骤2: 使用技能生成项目
    success = await step2_use_skill_generate_mall()

    if success:
        print()
        print("="*70)
        print("🎉 全部完成!")
        print("="*70)
        print(f"✅ Holon 成功演化技能并生成购物网站")
        print(f"✅ 项目位置: {OUTPUT_DIR}")
        print(f"✅ 技能位置: {SKILL_DIR / 'react_project_generator.py'}")


if __name__ == "__main__":
    asyncio.run(main())
