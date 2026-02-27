"""Environment parsing helpers adapted from Harborpilot backend patterns."""

from __future__ import annotations

import os
from typing import Sequence


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def parse_bool(value: object, *, default: bool = False) -> bool:
    """Parse a loose boolean value."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def parse_typed_value(value: str) -> str | int | float | bool:
    """Parse env-like scalar values into simple native types."""
    raw = str(value).strip()
    lower = raw.lower()
    if lower in _TRUE_VALUES:
        return True
    if lower in _FALSE_VALUES:
        return False
    if raw.isdigit():
        return int(raw)
    try:
        # Keep ints handled by `isdigit()` so "1" is not coerced to float.
        return float(raw)
    except ValueError:
        return raw


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return parse_bool(value, default=default)


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def env_list(name: str, default: Sequence[str] | None = None) -> list[str]:
    """Parse comma-separated env list values."""
    value = os.environ.get(name)
    if value is None:
        return list(default or [])
    parsed = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]
    return parsed or list(default or [])
