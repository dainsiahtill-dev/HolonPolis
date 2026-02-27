"""Repository Learner - 让 Holon 学习指定代码仓库。

Holon 可以：
1. 克隆/访问指定代码仓库
2. 分析代码结构、模式、最佳实践
3. 提取知识并转化为可复用的技能
4. 存储学习到的记忆中
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import structlog

from holonpolis.config import settings
from holonpolis.kernel.llm.llm_runtime import LLMConfig, LLMMessage, get_llm_runtime
from holonpolis.kernel.llm.provider_config import get_provider_manager
from holonpolis.services.memory_service import MemoryService

logger = structlog.get_logger()


@dataclass
class RepositoryAnalysis:
    """代码仓库分析结果。"""

    repo_url: str
    repo_name: str
    languages: Dict[str, int]  # 语言 -> 文件数
    total_files: int
    total_lines: int
    key_patterns: List[str]  # 识别的关键模式
    architecture: str  # 架构描述
    learnings: List[str]  # 学习到的知识


@dataclass
class LearningResult:
    """学习结果。"""

    success: bool
    holon_id: str
    repo_url: str
    analysis: Optional[RepositoryAnalysis] = None
    generated_skill: Optional[str] = None
    memories_created: int = 0
    error_message: Optional[str] = None


class RepositoryLearner:
    """代码仓库学习器。

    让 Holon 学习指定代码仓库，提取知识并转化为技能。
    """

    def __init__(self, holon_id: str, provider_id: Optional[str] = None):
        self.holon_id = holon_id
        self.memory = MemoryService(holon_id)
        self.runtime = get_llm_runtime()
        self.provider_manager = get_provider_manager()
        self.provider_id = provider_id or self._select_best_provider()

    def _select_best_provider(self) -> str:
        """选择最佳的代码分析 provider。"""
        providers = self.provider_manager.list_providers(mask_secrets=True)
        provider_ids = [p["provider_id"] for p in providers]

        # 优先选择本地 Ollama（稳定可用）
        if "ollama-local" in provider_ids:
            return "ollama-local"
        if "ollama" in provider_ids:
            return "ollama"
        if "kimi-coding" in provider_ids:
            return "kimi-coding"

        return provider_ids[0] if provider_ids else "openai"

    async def learn_repository(
        self,
        repo_url: str,
        branch: str = "main",
        depth: int = 3,
        focus_areas: Optional[List[str]] = None,
    ) -> LearningResult:
        """学习指定代码仓库。

        Args:
            repo_url: 仓库 URL (https://github.com/...)
            branch: 分支名
            depth: 分析深度 (1-5)
            focus_areas: 关注领域 (如 ["architecture", "testing", "patterns"])

        Returns:
            LearningResult
        """
        logger.info(
            "repository_learning_started",
            holon_id=self.holon_id,
            repo_url=repo_url,
            branch=branch,
        )

        try:
            # Phase 1: 获取仓库
            repo_path = await self._fetch_repository(repo_url, branch)
            if not repo_path:
                return LearningResult(
                    success=False,
                    holon_id=self.holon_id,
                    repo_url=repo_url,
                    error_message="Failed to fetch repository",
                )

            # Phase 2: 分析仓库结构
            analysis = await self._analyze_repository(repo_path, repo_url, depth)

            # Phase 3: 使用 LLM 深入学习
            deep_learnings = await self._deep_learning(repo_path, analysis, focus_areas)
            analysis.learnings.extend(deep_learnings)

            # Phase 4: 存储到 Holon 记忆
            memories_count = await self._store_learnings(analysis)

            # Phase 5: 生成技能代码（可选）
            generated_skill = await self._generate_skill_from_learning(analysis)

            logger.info(
                "repository_learning_completed",
                holon_id=self.holon_id,
                repo_url=repo_url,
                memories_created=memories_count,
            )

            return LearningResult(
                success=True,
                holon_id=self.holon_id,
                repo_url=repo_url,
                analysis=analysis,
                generated_skill=generated_skill,
                memories_created=memories_count,
            )

        except Exception as e:
            logger.error("repository_learning_failed", error=str(e), holon_id=self.holon_id)
            return LearningResult(
                success=False,
                holon_id=self.holon_id,
                repo_url=repo_url,
                error_message=str(e),
            )

    async def _fetch_repository(self, repo_url: str, branch: str) -> Optional[Path]:
        """获取仓库到临时目录。支持 URL 或本地路径（包括非 git 目录）。"""
        # 检查是否是本地路径
        local_path = Path(repo_url)
        if local_path.exists() and local_path.is_dir():
            # 支持非 git 目录的直接学习
            return local_path

        # 验证 URL
        parsed = urlparse(repo_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid repository URL: {repo_url}")

        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix="holon_repo_"))

        try:
            # 使用 git clone
            logger.info("cloning_repository", repo_url=repo_url, temp_dir=str(temp_dir))

            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(temp_dir / "repo")],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                # 尝试不指定分支
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(temp_dir / "repo")],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    logger.error("git_clone_failed", stderr=result.stderr)
                    return None

            return temp_dir / "repo"

        except subprocess.TimeoutExpired:
            logger.error("git_clone_timeout")
            return None
        except Exception as e:
            logger.error("git_clone_error", error=str(e))
            return None

    async def _analyze_repository(
        self, repo_path: Path, repo_url: str, depth: int
    ) -> RepositoryAnalysis:
        """分析仓库结构。"""
        logger.info("analyzing_repository", repo_path=str(repo_path))

        # 统计文件和语言
        languages: Dict[str, int] = {}
        total_files = 0
        total_lines = 0
        key_files: List[Path] = []

        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and self._should_analyze_file(file_path):
                total_files += 1
                lang = self._detect_language(file_path)
                languages[lang] = languages.get(lang, 0) + 1

                # 统计行数
                try:
                    lines = len(file_path.read_text(encoding="utf-8", errors="ignore").splitlines())
                    total_lines += lines
                except:
                    pass

                # 识别关键文件
                if self._is_key_file(file_path):
                    key_files.append(file_path.relative_to(repo_path))

                # 限制分析文件数量
                if total_files >= 1000:
                    break

        # 提取仓库名
        repo_name = repo_url.split("/")[-1].replace(".git", "")

        # 识别关键模式
        key_patterns = await self._extract_patterns(repo_path, key_files[:20])

        # 架构分析
        architecture = await self._analyze_architecture(repo_path, key_files[:10])

        return RepositoryAnalysis(
            repo_url=repo_url,
            repo_name=repo_name,
            languages=languages,
            total_files=total_files,
            total_lines=total_lines,
            key_patterns=key_patterns,
            architecture=architecture,
            learnings=[],
        )

    def _should_analyze_file(self, file_path: Path) -> bool:
        """判断是否应该分析该文件。"""
        # 忽略的文件
        ignore_patterns = [
            r"node_modules",
            r"\.git",
            r"__pycache__",
            r"\.venv",
            r"dist",
            r"build",
            r"\.idea",
            r"\.vscode",
            r"vendor",
        ]

        path_str = str(file_path)
        for pattern in ignore_patterns:
            if re.search(pattern, path_str):
                return False

        # 只分析代码文件
        code_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
            ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".r", ".m", ".mm", ".vue", ".svelte",
        }

        return file_path.suffix.lower() in code_extensions

    def _detect_language(self, file_path: Path) -> str:
        """检测文件语言。"""
        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React",
            ".tsx": "React/TS",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".vue": "Vue",
            ".svelte": "Svelte",
        }
        return ext_map.get(file_path.suffix.lower(), "Other")

    def _is_key_file(self, file_path: Path) -> bool:
        """判断是否为关键文件。"""
        key_names = [
            "main", "index", "app", "server", "client", "core",
            "config", "setup", "readme", "api", "router", "store",
        ]
        name = file_path.stem.lower()
        return any(key in name for key in key_names)

    async def _extract_patterns(self, repo_path: Path, key_files: List[Path]) -> List[str]:
        """提取代码模式。"""
        patterns = []

        # 检测常见模式
        pattern_files = {
            "factory": [r"factory", r"create.*instance"],
            "singleton": [r"singleton", r"getInstance"],
            "observer": [r"observer", r"event.*emit", r"subscribe"],
            "mvc": [r"model", r"view", r"controller"],
            "dependency_injection": [r"inject", r"@Injectable"],
            "middleware": [r"middleware", r"use\("],
        }

        for file_path in key_files[:10]:  # 限制分析文件数
            try:
                content = (repo_path / file_path).read_text(encoding="utf-8", errors="ignore")
                for pattern_name, regexes in pattern_files.items():
                    for regex in regexes:
                        if re.search(regex, content, re.IGNORECASE):
                            patterns.append(pattern_name)
                            break
            except:
                pass

        return list(set(patterns))

    async def _analyze_architecture(self, repo_path: Path, key_files: List[Path]) -> str:
        """分析架构。"""
        # 简单的架构推断
        has_src = (repo_path / "src").exists()
        has_lib = (repo_path / "lib").exists()
        has_tests = (repo_path / "tests").exists() or (repo_path / "test").exists()
        has_docs = (repo_path / "docs").exists()

        arch_parts = []
        if has_src:
            arch_parts.append("src-based")
        if has_lib:
            arch_parts.append("library structure")
        if has_tests:
            arch_parts.append("with tests")
        if has_docs:
            arch_parts.append("documented")

        return ", ".join(arch_parts) if arch_parts else "standard"

    async def _deep_learning(
        self,
        repo_path: Path,
        analysis: RepositoryAnalysis,
        focus_areas: Optional[List[str]],
    ) -> List[str]:
        """使用 LLM 进行深度学习。"""
        learnings = []

        # 准备上下文
        context = f"""
Repository: {analysis.repo_name}
Languages: {json.dumps(analysis.languages)}
Architecture: {analysis.architecture}
Patterns: {', '.join(analysis.key_patterns)}
"""

        # 读取一些示例代码
        sample_code = ""
        sample_count = 0
        for ext in [".ts", ".js", ".py", ".go"]:
            for file_path in repo_path.rglob(f"*{ext}"):
                if self._should_analyze_file(file_path) and sample_count < 3:
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        # 取前 50 行作为示例
                        lines = content.splitlines()[:50]
                        sample_code += f"\n// {file_path.name}\n" + "\n".join(lines) + "\n"
                        sample_count += 1
                    except:
                        pass

        # 构建提示
        focus_text = ""
        if focus_areas:
            focus_text = f"Focus on: {', '.join(focus_areas)}"

        prompt = f"""Analyze this code repository and extract key learnings:

{context}

Sample code:
```
{sample_code[:2000]}
```

{focus_text}

Provide 3-5 key learnings about:
1. Code organization patterns
2. Best practices used
3. Notable design decisions
4. How to apply these patterns

Format as concise bullet points."""

        # 调用 LLM
        provider_cfg = self.provider_manager.get_provider(self.provider_id)
        model = provider_cfg.model_specific.get("model", "qwen3-coder-30b-v12-q8-128k-dual3090:latest") if provider_cfg else "qwen3-coder-30b-v12-q8-128k-dual3090:latest"

        config = LLMConfig(
            provider_id=self.provider_id,
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )

        try:
            response = await self.runtime.chat(
                system_prompt="You are an expert code analyst. Extract actionable insights from code repositories.",
                user_prompt=prompt,
                config=config,
            )

            # 解析学习点
            content = response.content.strip()
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
                    learnings.append(line.lstrip("- *•12345. "))

        except Exception as e:
            logger.error("deep_learning_failed", error=str(e))
            learnings.append("Error during deep analysis")

        return learnings[:5]  # 限制学习点数量

    async def _store_learnings(self, analysis: RepositoryAnalysis) -> int:
        """存储学习到 Holon 记忆。"""
        count = 0

        # 存储基本信息
        await self.memory.remember(
            content=f"Learned repository: {analysis.repo_name} ({analysis.repo_url})",
            kind="fact",
            tags=["repository", "learning", analysis.repo_name],
            importance=0.9,
        )
        count += 1

        # 存储技术栈
        tech_stack = ", ".join([f"{lang}({count})" for lang, count in analysis.languages.items()])
        await self.memory.remember(
            content=f"{analysis.repo_name} tech stack: {tech_stack}",
            kind="fact",
            tags=["tech-stack", analysis.repo_name],
            importance=0.8,
        )
        count += 1

        # 存储架构信息
        await self.memory.remember(
            content=f"{analysis.repo_name} architecture: {analysis.architecture}",
            kind="pattern",
            tags=["architecture", analysis.repo_name],
            importance=0.85,
        )
        count += 1

        # 存储学习点
        for learning in analysis.learnings:
            await self.memory.remember(
                content=f"From {analysis.repo_name}: {learning}",
                kind="procedure",
                tags=["learning", "best-practice", analysis.repo_name],
                importance=0.8,
            )
            count += 1

        return count

    async def _generate_skill_from_learning(
        self, analysis: RepositoryAnalysis
    ) -> Optional[str]:
        """从学习生成技能代码（可选）。"""
        # 如果是 TypeScript/JavaScript 项目，生成对应的工具函数
        main_lang = max(analysis.languages, key=analysis.languages.get) if analysis.languages else ""

        if main_lang in ["TypeScript", "JavaScript"]:
            # 可以生成基于学习的工具函数
            return f"// Generated skill based on {analysis.repo_name}\n// TODO: Implement specific patterns"

        return None


class RepositoryLearningService:
    """仓库学习服务 - 供外部调用。"""

    def __init__(self):
        self._learners: Dict[str, RepositoryLearner] = {}

    def get_learner(self, holon_id: str) -> RepositoryLearner:
        """获取或创建 Learner。"""
        if holon_id not in self._learners:
            self._learners[holon_id] = RepositoryLearner(holon_id)
        return self._learners[holon_id]

    async def learn(
        self,
        holon_id: str,
        repo_url: str,
        branch: str = "main",
        depth: int = 3,
        focus_areas: Optional[List[str]] = None,
    ) -> LearningResult:
        """让指定 Holon 学习仓库。

        这是主要的对外接口。
        """
        learner = self.get_learner(holon_id)
        return await learner.learn_repository(repo_url, branch, depth, focus_areas)


def create_repository_learning_service() -> RepositoryLearningService:
    """工厂函数。"""
    return RepositoryLearningService()
