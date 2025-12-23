"""
Answer Generation API Route
Location: RAG/routes/answer.py

Handles RAG-based answer generation using retrieved context.
Generates answers with optional source citations.
"""

from typing import Literal, Optional
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai import OpenAI
from llama_index import ServiceContext
from RAG.core.config import qdrant_client, service_context, trace_decorator, li_callback_manager
from RAG.routes import search as search_module

router = APIRouter(tags=["Answer"])


class SearchResult(BaseModel):
    """Source result item from generated answer."""
    id: Optional[str] = None
    text: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


class AnswerRequest(BaseModel):
    """Request model for answer generation endpoint."""
    query: str
    mode: Literal["vector", "hybrid"] = "vector"
    top_k: int = 5
    alpha: Optional[float] = None
    prompt: Optional[str] = None
    include_sources: bool = True


class AnswerResponse(BaseModel):
    """Response model for answer generation endpoint."""
    mode: Literal["vector", "hybrid"]
    query: str
    top_k: int
    answer: str
    sources: Optional[list[SearchResult]] = None


@trace_decorator(name="answer")
@router.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    """
    Generate an answer using retrieved chunks + base prompt + query.
    
    Implements RAG (Retrieval-Augmented Generation) to provide 
    answer with optional sources.
    
    Parameters:
    -----------
    req : AnswerRequest
        Request containing query, mode, and generation options
        - mode="vector": dense-only with collection "hacker-news"
        - mode="hybrid": sparse+dense with collection "hacker-news-hybrid" 
          (alpha controls weighting)
    
    Returns:
    --------
    AnswerResponse
        Generated answer with optional source documents
        
    Raises:
    -------
    HTTPException
        If Qdrant is unavailable or answer generation fails
    """
    # Ensure indices
    eff_mode = req.mode
    try:
        search_module._ensure_indices(eff_mode)
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

    # Build per-request LLM with optional base prompt
    llm_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    base_prompt = req.prompt or (
        "You are a helpful RAG assistant. Read the retrieved context chunks and answer the user's question. "
        "Only use information from the provided chunks. If the answer is not contained, say you don't know."
    )
    llm = OpenAI(model=llm_model, system_prompt=base_prompt)
    per_request_sc = ServiceContext.from_defaults(
        embed_model=service_context.embed_model,
        llm=llm,
        callback_manager=li_callback_manager,
    )

    # Build query engine according to mode
    if req.mode == "vector":
        # Ensure collection exists
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
        
        # Get vector index from search module (late import to avoid circular imports)
        vector_index = search_module._VECTOR_INDEX
        qe = vector_index.as_query_engine(
            similarity_top_k=req.top_k,
            service_context=per_request_sc,
        )
    else:
        # hybrid
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
        
        # Get hybrid index from search module
        hybrid_index = search_module._HYBRID_INDEX
        effective_alpha = req.alpha if req.alpha is not None else 0.5
        qe = hybrid_index.as_query_engine(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            sparse_top_k=req.top_k,
            similarity_top_k=req.top_k,
            alpha=effective_alpha,
            service_context=per_request_sc,
        )

    # Query and build response
    try:
        resp = qe.query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    answer_text = str(resp)
    sources: Optional[list[SearchResult]] = None
    if req.include_sources:
        sources = []
        for s in getattr(resp, "source_nodes", []) or []:
            item: dict = {
                "text": getattr(getattr(s, "node", None), "text", "") or getattr(s, "text", ""),
            }
            node_obj = getattr(s, "node", None)
            node_id = (
                getattr(s, "node_id", None)
                or (getattr(node_obj, "node_id", None) if node_obj is not None else None)
                or (getattr(node_obj, "id_", None) if node_obj is not None else None)
            )
            if node_id:
                item["id"] = node_id
            score = getattr(s, "score", None)
            if score is not None:
                item["score"] = float(score)
            meta = getattr(node_obj, "metadata", None) if node_obj is not None else None
            if meta is not None:
                item["metadata"] = dict(meta)
            sources.append(SearchResult(**item))

    return AnswerResponse(mode=req.mode, query=req.query, top_k=req.top_k, answer=answer_text, sources=sources)
