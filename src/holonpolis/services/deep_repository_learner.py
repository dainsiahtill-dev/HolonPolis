"""Deep Repository Learner - 深度仓库学习器

逐文件学习，将每个文件内容存入向量记忆库
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import hashlib

import structlog

from holonpolis.services.memory_service import MemoryService

logger = structlog.get_logger()


@dataclass
class FileLearningResult:
    """单个文件的学习结果"""
    file_path: str
    language: str
    content_summary: str
    key_exports: List[str]
    memory_id: Optional[str] = None
    stored_successfully: bool = False


class DeepRepositoryLearner:
    """深度仓库学习器 - 逐文件学习"""

    # 代码文件扩展名映射
    LANGUAGE_MAP = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'React',
        '.ts': 'TypeScript',
        '.tsx': 'React/TypeScript',
        '.vue': 'Vue',
        '.go': 'Go',
        '.rs': 'Rust',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C/C++ Header',
        '.cs': 'C#',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'Sass',
        '.less': 'Less',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.xml': 'XML',
        '.sql': 'SQL',
        '.sh': 'Shell',
        '.bash': 'Bash',
        '.zsh': 'Zsh',
        '.ps1': 'PowerShell',
        '.dockerfile': 'Dockerfile',
        '.md': 'Markdown',
        '.txt': 'Text',
    }

    # 要忽略的文件和目录
    IGNORE_PATTERNS = [
        'node_modules',
        '.git',
        '__pycache__',
        '.venv',
        'venv',
        'dist',
        'build',
        '.idea',
        '.vscode',
        'vendor',
        'target',
        'out',
        'bin',
        'obj',
        '.next',
        '.nuxt',
        'coverage',
        '.coverage',
        '*.min.js',
        '*.min.css',
        '*.map',
        '*.lock',
    ]

    def __init__(self, holon_id: str):
        self.holon_id = holon_id
        self.memory = MemoryService(holon_id)
        self.stats = {
            'files_processed': 0,
            'files_stored': 0,
            'total_lines': 0,
            'languages': {},
        }

    def detect_language(self, file_path: Path) -> str:
        """检测文件语言"""
        ext = file_path.suffix.lower()

        # 特殊处理 Dockerfile
        if file_path.name.lower() == 'dockerfile':
            return 'Dockerfile'

        # 特殊处理 Makefile
        if file_path.name.lower() == 'makefile':
            return 'Makefile'

        return self.LANGUAGE_MAP.get(ext, 'Unknown')

    def should_ignore(self, file_path: Path) -> bool:
        """判断是否应该忽略该文件"""
        path_str = str(file_path)

        for pattern in self.IGNORE_PATTERNS:
            if pattern in path_str:
                return True
            if '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(path_str, pattern):
                    return True

        return False

    def extract_summary(self, content: str, max_length: int = 500) -> str:
        """提取文件内容摘要"""
        lines = content.split('\n')

        # 移除空行和注释行
        meaningful_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(('#', '//', '/*', '*', '<!--')):
                meaningful_lines.append(stripped)

        # 取前 N 行作为摘要
        summary = '\n'.join(meaningful_lines[:20])

        if len(summary) > max_length:
            summary = summary[:max_length] + '...'

        return summary

    def extract_exports(self, content: str, language: str) -> List[str]:
        """提取文件导出的内容"""
        exports = []

        if language in ['JavaScript', 'TypeScript', 'React', 'React/TypeScript']:
            # ES6 exports
            import re

            # export const/var/let/function/class
            patterns = [
                r'export\s+(?:const|let|var|function|class|interface|type)\s+(\w+)',
                r'export\s+\{\s*([^}]+)\s*\}',
                r'export\s+default\s+(?:class|function)?\s*(\w+)',
                r'module\.exports\s*=\s*\{?\s*(\w+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ''
                    if match:
                        exports.extend([e.strip() for e in match.split(',') if e.strip()])

        elif language == 'Python':
            import re
            # __all__ = ['func1', 'func2']
            all_pattern = r"__all__\s*=\s*\[(.*?)\]"
            match = re.search(all_pattern, content)
            if match:
                exports = [e.strip().strip('"\'') for e in match.group(1).split(',')]

            # def func_name(
            func_pattern = r'^def\s+(\w+)\s*\('
            exports.extend(re.findall(func_pattern, content, re.MULTILINE))

            # class ClassName
            class_pattern = r'^class\s+(\w+)'
            exports.extend(re.findall(class_pattern, content, re.MULTILINE))

        return list(set(exports))[:10]  # 限制数量

    async def learn_file(self, file_path: Path, repo_name: str) -> FileLearningResult:
        """学习单个文件"""
        try:
            # 读取文件
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            # 检测语言
            language = self.detect_language(file_path)

            # 提取摘要
            summary = self.extract_summary(content)

            # 提取导出
            exports = self.extract_exports(content, language)

            # 统计行数
            lines_count = len(content.split('\n'))
            self.stats['total_lines'] += lines_count

            # 更新语言统计
            self.stats['languages'][language] = self.stats['languages'].get(language, 0) + 1

            # 构建记忆内容
            relative_path = str(file_path.relative_to(file_path.parent.parent))

            memory_content = f"""File: {relative_path} (from {repo_name})
Language: {language}
Lines: {lines_count}

Exports: {', '.join(exports) if exports else 'None'}

Summary:
{summary}"""

            # 存储到记忆
            memory_id = await self.memory.remember(
                content=memory_content,
                kind='fact',
                tags=['code-file', language.lower(), repo_name, relative_path],
                importance=0.7,
            )

            self.stats['files_stored'] += 1

            return FileLearningResult(
                file_path=str(file_path),
                language=language,
                content_summary=summary[:200],
                key_exports=exports,
                memory_id=memory_id,
                stored_successfully=True,
            )

        except Exception as e:
            logger.warning("file_learning_failed", file=str(file_path), error=str(e))
            return FileLearningResult(
                file_path=str(file_path),
                language='Unknown',
                content_summary='',
                key_exports=[],
                stored_successfully=False,
            )

    async def learn_repository(
        self,
        repo_path: Path,
        repo_name: str,
        max_files: int = 500,
    ) -> Dict:
        """学习整个仓库的所有文件"""
        logger.info(
            "deep_repository_learning_started",
            holon_id=self.holon_id,
            repo_path=str(repo_path),
            repo_name=repo_name,
        )

        # 收集所有代码文件
        code_files: List[Path] = []

        for ext in self.LANGUAGE_MAP.keys():
            for file_path in repo_path.rglob(f'*{ext}'):
                if not self.should_ignore(file_path):
                    code_files.append(file_path)

        # 特殊文件
        special_files = ['Dockerfile', 'Makefile', 'README.md', 'package.json']
        for special in special_files:
            for file_path in repo_path.rglob(special):
                if not self.should_ignore(file_path) and file_path not in code_files:
                    code_files.append(file_path)

        # 限制文件数量
        if len(code_files) > max_files:
            logger.warning("too_many_files", total=len(code_files), max=max_files)
            # 优先保留重要文件
            code_files = sorted(code_files, key=lambda p: self._file_priority(p))[:max_files]

        logger.info("files_to_learn", count=len(code_files))

        # 逐个学习文件
        results: List[FileLearningResult] = []
        self.stats['files_processed'] = len(code_files)

        for i, file_path in enumerate(code_files, 1):
            if i % 50 == 0:
                logger.info("learning_progress", current=i, total=len(code_files))

            result = await self.learn_file(file_path, repo_name)
            if result.stored_successfully:
                results.append(result)

        # 存储仓库概览
        await self._store_repository_overview(repo_name, results)

        logger.info(
            "deep_repository_learning_completed",
            holon_id=self.holon_id,
            files_processed=self.stats['files_processed'],
            files_stored=self.stats['files_stored'],
        )

        return {
            'success': True,
            'repo_name': repo_name,
            'files_processed': self.stats['files_processed'],
            'files_stored': self.stats['files_stored'],
            'total_lines': self.stats['total_lines'],
            'languages': self.stats['languages'],
            'sample_files': [r.file_path for r in results[:5]],
        }

    def _file_priority(self, file_path: Path) -> int:
        """计算文件优先级（用于排序）"""
        name = file_path.name.lower()
        path_str = str(file_path).lower()

        # 高优先级
        if any(x in name for x in ['index', 'main', 'app', 'core', 'config']):
            return 0

        # 中等优先级 - 源代码
        if file_path.suffix in ['.tsx', '.ts', '.jsx', '.js', '.vue', '.py']:
            return 1

        # 较低优先级
        return 2

    async def _store_repository_overview(self, repo_name: str, results: List[FileLearningResult]):
        """存储仓库概览信息"""
        # 技术栈概览
        top_languages = sorted(
            self.stats['languages'].items(),
            key=lambda x: -x[1]
        )[:5]

        overview = f"""Repository: {repo_name}

Total Files: {self.stats['files_processed']}
Files Learned: {self.stats['files_stored']}
Total Lines: {self.stats['total_lines']}

Tech Stack:
{chr(10).join(f"- {lang}: {count} files" for lang, count in top_languages)}

Key Components:
{chr(10).join(f"- {r.file_path} ({r.language})" for r in results[:10] if r.key_exports)}"""

        await self.memory.remember(
            content=overview,
            kind='pattern',
            tags=['repository-overview', repo_name],
            importance=0.9,
        )


# 便捷函数
async def learn_repository_deep(
    holon_id: str,
    repo_path: Path,
    repo_name: Optional[str] = None,
) -> Dict:
    """便捷函数：深度学习仓库"""
    if repo_name is None:
        repo_name = repo_path.name

    learner = DeepRepositoryLearner(holon_id)
    return await learner.learn_repository(repo_path, repo_name)
