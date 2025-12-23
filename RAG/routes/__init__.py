"""
Routes module for RAG API.
Contains all API endpoint routers.
"""

from .collections import router as collections_router
from .search import router as search_router
from .answer import router as answer_router

__all__ = [
    "collections_router",
    "search_router",
    "answer_router",
]
