import os
from typing import Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, HnswConfig

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# Load env
load_dotenv()

# Configure embeddings once
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")

# Qdrant client
_qdrant = QdrantClient(
    os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)

def _build_indices():
    """Build indices lazily to avoid startup failure when Qdrant is unreachable."""
    vector_store = QdrantVectorStore(
        client=_qdrant,
        collection_name="hacker-news",
    )
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    hybrid_store = QdrantVectorStore(
        client=_qdrant,
        collection_name="hacker-news-hybrid",
        enable_hybrid=True,
    )
    hybrid_index = VectorStoreIndex.from_vector_store(vector_store=hybrid_store)

    return vector_index, hybrid_index

# Built on first request
_VECTOR_INDEX = None
_HYBRID_INDEX = None

app = FastAPI(title="RAG Search API", version="0.1.0")


class SearchResponse(BaseModel):
    mode: Literal["vector", "hybrid"]
    query: str
    top_k: int
    results: list[dict]


class SearchResult(BaseModel):
    id: Optional[str] = None
    text: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/collections/init")
def init_collections(
    dense_collection: str = "hacker-news",
    hybrid_collection: str = "hacker-news-hybrid",
    dim: int = 1024,
    distance: str = "Cosine",
) -> dict:
    """
    Create/check Qdrant collections used by the API.
    - dense_collection: standard dense-only collection.
    - hybrid_collection: sparse+dense enabled for hybrid queries.
    - dim: embedding dimension for BAAI/bge-large-en (1024).
    - distance: metric for dense vectors.
    """
    from fastapi import HTTPException

    if _qdrant is None or os.environ.get("QDRANT_URL") in (None, ""):
        raise HTTPException(
            status_code=503,
            detail=(
                "Qdrant client not configured. Set QDRANT_URL (and QDRANT_API_KEY for Cloud) "
                "or run local Docker at http://localhost:6333."
            ),
        )
    try:
        # Quick connectivity check
        _qdrant.get_collections()
        # Dense collection
        if not _qdrant.collection_exists(dense_collection):
            _qdrant.create_collection(
                dense_collection,
                vectors_config=VectorParams(
    size=dim,
    distance=Distance[distance.upper()],
),
                hnsw_config=HnswConfig(m=16, ef_construct=200),
            )
        # Hybrid collection with sparse enabled
        if not _qdrant.collection_exists(hybrid_collection):
            _qdrant.create_collection(
                hybrid_collection,
                vectors_config=VectorParams(size=dim, distance=Distance[distance]),
                sparse_vectors_config=SparseVectorParams(),
                hnsw_config=HnswConfig(m=16, ef_construct=200),
            )
        return {"status": "initialized", "dense": dense_collection, "hybrid": hybrid_collection}
    except Exception as e:
        # Provide actionable guidance for common 401/403/connection errors
        msg = (
            "Failed to init collections: "
            f"{e}. Ensure Qdrant is reachable and credentials are correct. "
            "Set QDRANT_URL (and QDRANT_API_KEY for Cloud) or run local Docker at http://localhost:6333."
        )
        raise HTTPException(status_code=503, detail=msg)


def _ensure_indices(request_mode: str) -> None:
    """Initialize only the indices needed for the request mode."""
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
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Failed to init hybrid index. "
                        f"Details: {e}. If error mentions FastEmbed, install it: `pip install fastembed`."
                    ),
                )


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="User query"),
    mode: Literal["vector", "hybrid"] = Query("vector", description="Search mode"),
    top_k: int = Query(5, ge=1, le=25, description="Number of results to return"),
    alpha: Optional[float] = Query(None, ge=0.0, le=1.0, description="Hybrid weight: 0=sparse, 1=dense"),
    include_scores: bool = Query(False, description="Include similarity scores in results"),
    include_metadata: bool = Query(False, description="Include node metadata in results"),
) -> SearchResponse:
    """
    Perform semantic retrieval.
    - mode="vector": dense-only semantic search from collection "hacker-news"
    - mode="hybrid": combined sparse+dense search from collection "hacker-news-hybrid"
      Optional `alpha` controls weighting (0.0 sparse ↔ 1.0 dense)
    """
    # Ensure only the needed indices are initialized
    try:
        _ensure_indices(mode)
    except Exception as e:
        from fastapi import HTTPException
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
            if not _qdrant.collection_exists("hacker-news"):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection 'hacker-news' not found. "
                        "Call POST /collections/init to create required collections."
                    ),
                )
        except Exception:
            # If Qdrant client errors, let later handling surface connectivity guidance
            pass
        retriever = VectorIndexRetriever(
            index=_VECTOR_INDEX,
            similarity_top_k=top_k,
        )
        try:
            nodes = retriever.retrieve(query)
        except Exception as e:
            from fastapi import HTTPException
            # Common case: collection missing -> Qdrant 404
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
            if not _qdrant.collection_exists("hacker-news-hybrid"):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection 'hacker-news-hybrid' not found. "
                        "Call POST /collections/init to create required collections."
                    ),
                )
        except Exception:
            pass
        retriever = VectorIndexRetriever(
            index=_HYBRID_INDEX,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            sparse_top_k=top_k,
            similarity_top_k=top_k,
            alpha=alpha if alpha is not None else 0.5,
        )
        try:
            nodes = retriever.retrieve(query)
        except Exception as e:
            from fastapi import HTTPException
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
        node_id = getattr(n, "node_id", None) or getattr(getattr(n, "node", None), "node_id", None) or getattr(getattr(n, "node", None), "id_", None)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
