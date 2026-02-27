"""React Project Generator - 通用 React 项目生成器。

设计原则：
1. 不包含任何硬编码的业务代码
2. 所有文件内容通过 LLM 动态生成
3. 只提供项目结构定义和文件组织框架
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from holonpolis.kernel.llm.llm_runtime import LLMConfig, get_llm_runtime
from holonpolis.kernel.llm.provider_config import get_provider_manager

logger = structlog.get_logger()


@dataclass
class ProjectFile:
    """项目文件定义。"""
    path: str
    content: str
    file_type: str = "code"


@dataclass
class ReactProjectBlueprint:
    """React 项目蓝图。"""
    project_name: str
    description: str
    files: List[ProjectFile]
    dependencies: Dict[str, str]
    dev_dependencies: Dict[str, str]
    scripts: Dict[str, str]


class ReactProjectGenerator:
    """React 前端项目生成器 - 完全由 LLM 驱动。

    原则：
    - 零硬编码业务代码
    - 所有内容由 LLM 动态生成
    - 仅提供结构框架
    """

    def __init__(self, provider_id: Optional[str] = None):
        self.runtime = get_llm_runtime()
        self.provider_manager = get_provider_manager()
        self.provider_id = provider_id or self._select_best_provider()

    def _select_best_provider(self) -> str:
        """选择最佳的代码生成 provider。"""
        providers = self.provider_manager.list_providers(mask_secrets=True)
        provider_ids = [p["provider_id"] for p in providers]

        if "kimi-coding" in provider_ids:
            return "kimi-coding"
        if "ollama-local" in provider_ids:
            return "ollama-local"
        if "ollama" in provider_ids:
            return "ollama"

        return provider_ids[0] if provider_ids else "openai_compat"

    async def generate_react_project(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
        style_theme: Optional[Dict[str, Any]] = None,
    ) -> ReactProjectBlueprint:
        """生成 React 项目 - 所有内容由 LLM 动态生成。"""
        logger.info(
            "react_project_generation_started",
            project_name=project_name,
            provider=self.provider_id,
        )

        all_files = []

        # 1. 生成配置文件 (LLM 驱动)
        config_files = await self._generate_config_files(project_name, description, requirements)
        all_files.extend(config_files)

        # 2. 生成源代码文件 (LLM 驱动)
        src_files = await self._generate_source_files(project_name, description, requirements)
        all_files.extend(src_files)

        # 提取依赖
        deps, dev_deps, scripts = self._extract_dependencies(requirements)

        return ReactProjectBlueprint(
            project_name=project_name,
            description=description,
            files=all_files,
            dependencies=deps,
            dev_dependencies=dev_deps,
            scripts=scripts,
        )

    async def _generate_config_files(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
    ) -> List[ProjectFile]:
        """生成配置文件 - 全部通过 LLM。"""
        files = []

        # package.json - LLM 生成
        pkg_prompt = f"""Generate package.json for: {project_name}
Description: {description}
Requirements: {requirements[:5]}

Output ONLY valid JSON, no markdown."""

        pkg_content = await self._llm_call(
            "You generate valid package.json files.",
            pkg_prompt,
        )
        files.append(ProjectFile("package.json", self._extract_code(pkg_content), "config"))

        # tsconfig.json - LLM 生成
        tsconfig_prompt = """Generate tsconfig.json for React + TypeScript + Vite project.
Requirements: strict mode, path aliases @/*, ES2020 target

Output ONLY JSON."""

        tsconfig_content = await self._llm_call(
            "You generate TypeScript configurations.",
            tsconfig_prompt,
        )
        files.append(ProjectFile("tsconfig.json", self._extract_code(tsconfig_content), "config"))

        # vite.config.ts - LLM 生成
        vite_prompt = """Generate vite.config.ts for React TypeScript project with @/ alias.

Output ONLY TypeScript code."""

        vite_content = await self._llm_call(
            "You generate Vite configurations.",
            vite_prompt,
        )
        files.append(ProjectFile("vite.config.ts", self._extract_code(vite_content), "config"))

        # index.html - LLM 生成
        html_prompt = f"""Generate index.html for: {project_name}
Include viewport meta and title.

Output ONLY HTML."""

        html_content = await self._llm_call(
            "You generate HTML files.",
            html_prompt,
        )
        files.append(ProjectFile("index.html", self._extract_code(html_content), "config"))

        return files

    async def _generate_source_files(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
    ) -> List[ProjectFile]:
        """生成源代码文件 - 全部通过 LLM。"""
        files = []

        # 解析需要的文件结构
        structure = self._parse_structure(requirements)

        # 生成 main.tsx
        main_prompt = f"""Generate src/main.tsx for React app.
Project: {project_name}
Requirements: {requirements[:3]}

Output ONLY TypeScript React code."""

        main_content = await self._llm_call(
            "You are a React expert. Generate entry point code.",
            main_prompt,
        )
        files.append(ProjectFile("src/main.tsx", self._extract_code(main_content), "code"))

        # 生成 App.tsx
        app_prompt = f"""Generate src/App.tsx with routing.
Project: {project_name}
Description: {description}
Pages: {structure.get('pages', ['Home'])}

Output ONLY TypeScript React code."""

        app_content = await self._llm_call(
            "You are a React routing expert.",
            app_prompt,
        )
        files.append(ProjectFile("src/App.tsx", self._extract_code(app_content), "code"))

        # 生成页面
        for page in structure.get("pages", ["Home"]):
            page_prompt = f"""Generate src/pages/{page}.tsx
Project: {project_name}
Description: {description}
Requirements: {requirements[:5]}

Output ONLY TypeScript React code."""

            page_content = await self._llm_call(
                "You are a React page developer.",
                page_prompt,
            )
            files.append(ProjectFile(f"src/pages/{page}.tsx", self._extract_code(page_content), "code"))

        # 生成组件
        for component in structure.get("components", []):
            comp_prompt = f"""Generate src/components/{component}.tsx
Project: {project_name}
Requirements: {requirements[:3]}

Output ONLY TypeScript React code."""

            comp_content = await self._llm_call(
                "You are a React component developer.",
                comp_prompt,
            )
            files.append(ProjectFile(f"src/components/{component}.tsx", self._extract_code(comp_content), "code"))

        # 生成样式
        css_prompt = f"""Generate src/index.css
Project: {project_name}
Description: {description}
Requirements: {requirements[:3]}

Output ONLY CSS code."""

        css_content = await self._llm_call(
            "You are a CSS expert.",
            css_prompt,
        )
        files.append(ProjectFile("src/index.css", self._extract_code(css_content), "style"))

        return files

    def _parse_structure(self, requirements: List[str]) -> Dict[str, List[str]]:
        """从需求解析项目结构。"""
        structure = {"pages": [], "components": []}

        for req in requirements:
            # 简单解析页面和组件名称
            words = req.split()
            for i, word in enumerate(words):
                if word == "page" and i > 0:
                    name = words[i-1].strip(",;:.[]")
                    if name and name[0].isupper():
                        structure["pages"].append(name)
                if word == "component" and i > 0:
                    name = words[i-1].strip(",;:.[]")
                    if name and name[0].isupper():
                        structure["components"].append(name)

        # 默认值
        if not structure["pages"]:
            structure["pages"] = ["Home"]

        return structure

    async def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM。"""
        provider_cfg = self.provider_manager.get_provider(self.provider_id)
        model = provider_cfg.model if provider_cfg else "qwen2.5-coder:14b"

        config = LLMConfig(
            provider_id=self.provider_id,
            model=model,
            temperature=0.3,
            max_tokens=8000,
        )

        response = await self.runtime.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        return response.content

    def _extract_code(self, content: str) -> str:
        """提取代码，移除 markdown。"""
        code = content.strip()
        prefixes = ["```typescript", "```tsx", "```json", "```html", "```css", "```js", "```"]
        for prefix in prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):]
                break
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def _extract_dependencies(
        self, requirements: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """提取依赖。"""
        deps = {"react": "^18.3.1", "react-dom": "^18.3.1"}
        dev_deps = {"typescript": "^5.2.2", "vite": "^5.3.4"}
        scripts = {"dev": "vite", "build": "tsc && vite build"}

        req_text = " ".join(requirements).lower()
        if "router" in req_text:
            deps["react-router-dom"] = "^6.26.0"
        if "tailwind" in req_text:
            dev_deps["tailwindcss"] = "^3.4.0"

        return deps, dev_deps, scripts
