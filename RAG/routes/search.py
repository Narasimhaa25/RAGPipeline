"""
Search API Route
Location: RAG/routes/search.py

Handles semantic and hybrid search functionality.
Provides endpoints for vector-based and hybrid retrieval.
"""

from typing import Literal, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from RAG.core.config import qdrant_client, service_context, trace_decorator

router = APIRouter(tags=["Search"])

# Global indices (built on demand)
_VECTOR_INDEX = None
_HYBRID_INDEX = None


class SearchResult(BaseModel):
    """Search result item with text, metadata, and score."""
    id: Optional[str] = None
    text: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    mode: Literal["vector", "hybrid"]
    query: str
    top_k: int
    results: list[dict]


def _build_indices():
    """
    Build indices lazily to avoid startup failure when Qdrant is unreachable.
    Creates both vector and hybrid indices.
    
    Returns:
    --------
    tuple
        (vector_index, hybrid_index)
    """
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="hacker-news",
    )
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=service_context,
    )

    hybrid_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="hacker-news-hybrid",
        enable_hybrid=True,
    )
    hybrid_index = VectorStoreIndex.from_vector_store(
        vector_store=hybrid_store,
        service_context=service_context,
    )

    return vector_index, hybrid_index


def _ensure_indices(request_mode: str) -> None:
    """
    Initialize only the indices needed for the request mode.
    
    Args:
    ----
    request_mode : str
        Either "vector" or "hybrid"
        
    Raises:
    -------
    HTTPException
        If index initialization fails
    """
    global _VECTOR_INDEX, _HYBRID_INDEX
    if request_mode == "vector":
        if _VECTOR_INDEX is None:
            _VECTOR_INDEX, _ = _build_indices()
    else:
        # HYBRID: requires sparse encoder (FastEmbed)
        if _HYBRID_INDEX is None:
            try:
                _, _HYBRID_INDEX = _build_indices()
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Failed to init hybrid index. "
                        f"Details: {e}. If error mentions FastEmbed, install it: `pip install fastembed`."
                    ),
                )


@trace_decorator(name="search")
@router.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="User query"),
    mode: Literal["vector", "hybrid"] = Query("vector", description="Search mode"),
    top_k: int = Query(5, ge=1, le=25, description="Number of results to return"),
    alpha: Optional[float] = Query(None, ge=0.0, le=1.0, description="Hybrid weight: 0=sparse, 1=dense"),
    include_scores: bool = Query(False, description="Include similarity scores in results"),
    include_metadata: bool = Query(False, description="Include node metadata in results"),
) -> SearchResponse:
    """
    Perform semantic retrieval with optional hybrid search.
    
    Parameters:
    -----------
    query : str
        User search query
    mode : str
        "vector" for dense-only or "hybrid" for sparse+dense search
    top_k : int
        Number of results to return (1-25)
    alpha : float, optional
        Weight for hybrid search (0.0=sparse, 1.0=dense)
    include_scores : bool
        Whether to include similarity scores
    include_metadata : bool
        Whether to include document metadata
    
    Returns:
    --------
    SearchResponse
        Results with query metadata and retrieved documents
    """
    # Ensure only the needed indices are initialized
    try:
        _ensure_indices(mode)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Qdrant unavailable or misconfigured. "
                f"Details: {e}. "
                "Set QDRANT_URL (and QDRANT_API_KEY for Cloud) or run local Docker at http://localhost:6333. "
                "Call POST /collections/init to create required collections."
            ),
        )

    if mode == "vector":
        # Proactively ensure dense collection exists
        try:
            if not qdrant_client.collection_exists("hacker-news"):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection 'hacker-news' not found. "
                        "Call POST /collections/init to create required collections."
                    ),
                )
        except Exception:
            pass
        retriever = _VECTOR_INDEX.as_retriever(
            similarity_top_k=top_k,
        )
        try:
            nodes = retriever.retrieve(query)
        except Exception as e:
            msg = str(e)
            if "Collection `hacker-news`" in msg and "doesn't exist" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection 'hacker-news' not found. "
                        "Call POST /collections/init to create required collections."
                    ),
                )
            raise
    else:
        # Proactively ensure hybrid collection exists
        try:
            if not qdrant_client.collection_exists("hacker-news-hybrid"):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection 'hacker-news-hybrid' not found. "
                        "Call POST /collections/init to create required collections."
                    ),
                )
        except Exception:
            pass
        retriever = _HYBRID_INDEX.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            sparse_top_k=top_k,
            similarity_top_k=top_k,
            alpha=alpha if alpha is not None else 0.5,
        )
        try:
            nodes = retriever.retrieve(query)
        except Exception as e:
            msg = str(e)
            if "Collection `hacker-news-hybrid`" in msg and "doesn't exist" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection 'hacker-news-hybrid' not found. "
                        "Call POST /collections/init to create required collections."
                    ),
                )
            raise

    results: list[dict] = []
    for n in nodes:
        item: dict = {
            "text": getattr(n, "text", None) or getattr(getattr(n, "node", None), "text", ""),
        }
        # Attempt to include a stable id if available
        node_id = (
            getattr(n, "node_id", None)
            or getattr(getattr(n, "node", None), "node_id", None)
            or getattr(getattr(n, "node", None), "id_", None)
        )
        if node_id:
            item["id"] = node_id
        if include_scores:
            score = getattr(n, "score", None)
            if score is not None:
                item["score"] = float(score)
        if include_metadata:
            meta = None
            node_obj = getattr(n, "node", None)
            if node_obj is not None:
                meta = getattr(node_obj, "metadata", None)
            if meta is not None:
                item["metadata"] = dict(meta)
        results.append(item)

    return SearchResponse(mode=mode, query=query, top_k=top_k, results=results)
