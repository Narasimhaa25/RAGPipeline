"""
RAG Search API - Main Application Entry Point
Location: main.py (in project root or RAG/ directory)

Integrates all API modules: collections, search, and answer generation.
This is the primary entry point for running the FastAPI application.

To run:
    python main.py
    or
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import warnings

# Suppress pkg_resources deprecation warning from llama_index
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from fastapi import FastAPI

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import routers from RAG.routes
from RAG.routes import (
    collections_router,
    search_router,
    answer_router,
)

# Create FastAPI application instance
app = FastAPI(
    title="RAG Search API",
    version="0.1.0",
    description="A modular Retrieval-Augmented Generation (RAG) API with semantic search and answer generation",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# -------------------------
# Health Check Endpoint
# -------------------------

@app.get("/health", tags=["Health"])
def health() -> dict:
    """
    Health check endpoint to verify API is running.
    
    Returns:
    --------
    dict
        Status indicator
    """
    return {"status": "ok", "version": "0.1.0"}


# -------------------------
# Include Routers
# -------------------------

# Collections API - manages Qdrant collections
app.include_router(collections_router)

# Search API - semantic and hybrid search
app.include_router(search_router)

# Answer API - RAG-based answer generation
app.include_router(answer_router)


# -------------------------
# Root Endpoint
# -------------------------

@app.get("/", tags=["Info"])
def root() -> dict:
    """
    Root endpoint providing API information.
    
    Returns:
    --------
    dict
        API metadata and available endpoints
    """
    return {
        "name": "RAG Search API",
        "version": "0.1.0",
        "description": "Retrieval-Augmented Generation API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "collections": "/collections/init",
            "search": "/search",
            "answer": "/answer",
        }
    }


# -------------------------
# Application Startup
# -------------------------

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")
    
    # Run the application
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
