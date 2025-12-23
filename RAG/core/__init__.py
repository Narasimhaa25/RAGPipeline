"""
Core module for RAG API.
Contains shared configuration, clients, and utilities.
"""

from .config import (
    qdrant_client,
    service_context,
    trace_decorator,
    li_callback_manager,
)

__all__ = [
    "qdrant_client",
    "service_context",
    "trace_decorator",
    "li_callback_manager",
]
