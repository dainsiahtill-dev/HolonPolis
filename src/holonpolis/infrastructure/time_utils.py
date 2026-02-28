"""Time utility helpers.

Use timezone-aware UTC consistently across the project.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return current timezone-aware UTC datetime as ISO string."""
    return utc_now().isoformat()
