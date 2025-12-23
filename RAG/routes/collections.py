"""
Collections API Route
Location: RAG/routes/collections.py

Handles initialization and management of Qdrant collections.
Provides endpoints for creating and checking vector collections.
"""

from fastapi import APIRouter, HTTPException
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, HnswConfig
import os
from RAG.core.config import qdrant_client

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.post("/init")
def init_collections(
    dense_collection: str = "hacker-news",
    hybrid_collection: str = "hacker-news-hybrid",
    dim: int = 1024,
    distance: str = "Cosine",
) -> dict:
    """
    Create/check Qdrant collections used by the API.
    
    Parameters:
    -----------
    dense_collection : str
        Name of standard dense-only collection
    hybrid_collection : str
        Name of sparse+dense enabled collection for hybrid queries
    dim : int
        Embedding dimension for BAAI/bge-large-en (default: 1024)
    distance : str
        Distance metric for dense vectors (default: "Cosine")
    
    Returns:
    --------
    dict
        Status of initialization and collection names
        
    Raises:
    -------
    HTTPException
        If Qdrant client is not configured or collection creation fails
    """
    if qdrant_client is None or os.environ.get("QDRANT_URL") in (None, ""):
        raise HTTPException(
            status_code=503,
            detail=(
                "Qdrant client not configured. Set QDRANT_URL (and QDRANT_API_KEY for Cloud) "
                "or run local Docker at http://localhost:6333."
            ),
        )
    try:
        # Quick connectivity check
        qdrant_client.get_collections()
        
        # Dense collection
        if not qdrant_client.collection_exists(dense_collection):
            qdrant_client.create_collection(
                dense_collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance[distance.upper()],
                ),
                hnsw_config=HnswConfig(m=16, ef_construct=200),
            )
        
        # Hybrid collection with sparse enabled
        if not qdrant_client.collection_exists(hybrid_collection):
            qdrant_client.create_collection(
                hybrid_collection,
                vectors_config=VectorParams(size=dim, distance=Distance[distance.upper()]),
                sparse_vectors_config=SparseVectorParams(),
                hnsw_config=HnswConfig(m=16, ef_construct=200),
            )
        
        return {
            "status": "initialized",
            "dense": dense_collection,
            "hybrid": hybrid_collection
        }
    except Exception as e:
        msg = (
            "Failed to init collections: "
            f"{e}. Ensure Qdrant is reachable and credentials are correct. "
            "Set QDRANT_URL (and QDRANT_API_KEY for Cloud) or run local Docker at http://localhost:6333."
        )
        raise HTTPException(status_code=503, detail=msg)
