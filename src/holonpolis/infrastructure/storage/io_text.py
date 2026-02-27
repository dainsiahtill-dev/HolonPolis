"""Text and JSON file utilities for HolonPolis."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List


def _fsync_enabled() -> bool:
    """Check if fsync is enabled for atomic writes."""
    value = os.environ.get("HOLONPOLIS_IO_FSYNC", "strict").strip().lower()
    return value not in ("0", "false", "no", "off", "relaxed", "skip", "disabled")


def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_text_atomic(path: str, text: str) -> None:
    """Write text file atomically using temp file and replace."""
    if not path:
        return
    ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text or "")
        handle.flush()
        if _fsync_enabled():
            os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Write JSON file atomically."""
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    write_text_atomic(path, payload + "\n")


def is_run_artifact(rel_path: str) -> bool:
    """Check if path is a run artifact."""
    lowered = rel_path.lower().replace("\\", "/")
    if lowered.endswith("director_result.json") or lowered.endswith("director.result.json"):
        return True
    if lowered.endswith("events.jsonl") or lowered.endswith("runtime.events.jsonl"):
        return True
    if lowered.endswith("trajectory.json"):
        return True
    if lowered.endswith("qa_response.md") or lowered.endswith("qa.review.md"):
        return True
    if lowered.endswith("planner_response.md") or lowered.endswith("planner.output.md"):
        return True
    if lowered.endswith("ollama_response.md") or lowered.endswith("director_llm.output.md"):
        return True
    if lowered.endswith("reviewer_response.md") or lowered.endswith("auditor.review.md"):
        return True
    if lowered.endswith("runlog.md") or lowered.endswith("director.runlog.md"):
        return True
    return False


def _decode_text_bytes(data: bytes) -> str:
    """Decode bytes to text with fallback encodings."""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    if text:
        bad = text.count("\ufffd")
        if bad / max(len(text), 1) < 0.02:
            return text
    for enc in ("utf-8-sig", "gbk", "cp936"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def read_file_safe(path: str) -> str:
    """Safely read file contents with encoding fallback."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        return _decode_text_bytes(data)
    except Exception:
        return ""


def read_json_safe(path: str) -> Optional[Dict[str, Any]]:
    """Safely read JSON file."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def extract_field(text: str, patterns: List[str]) -> str:
    """Extract field from text using regex patterns."""
    if not text:
        return ""
    for pattern in patterns:
        try:
            match = re.search(pattern, text, flags=re.MULTILINE)
        except re.error:
            match = None
        if match:
            return match.group(1).strip()
    return ""


def format_mtime(path: str) -> str:
    """Format file modification time."""
    if not path or not os.path.exists(path):
        return "missing"
    try:
        from datetime import datetime
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def build_file_status(entries: List[tuple[str, str]]) -> List[str]:
    """Build file status lines."""
    lines: List[str] = []
    for label, path in entries:
        mtime = format_mtime(path)
        lines.append(f"{label}: {mtime}")
    return lines
