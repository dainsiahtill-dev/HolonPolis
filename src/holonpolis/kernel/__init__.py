"""Kernel layer - immutable physical laws of the system."""

from .lancedb.lancedb_factory import LanceDBFactory, get_lancedb_factory

__all__ = ["LanceDBFactory", "get_lancedb_factory"]
