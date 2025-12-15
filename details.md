# Vector vs Hybrid Search — What’s happening here

This project demonstrates two retrieval modes over your data using LlamaIndex + Qdrant:
- Vector (dense) search
- Hybrid (sparse + dense) search

It also includes a small FastAPI service that exposes these via a `/search` HTTP endpoint and a `/collections/init` endpoint to create the required Qdrant collections.

## Vector vs Hybrid
- Vector (dense) search:
  - Uses a neural embedding model (`BAAI/bge-large-en`) to convert text into 1024‑dimensional vectors.
  - Similarity: cosine distance in Qdrant.
  - Strengths: semantic matching (captures meaning beyond exact words), robust to paraphrasing.
  - Limitations: can miss rare terms, acronyms, or out‑of‑vocabulary words (vocabulary mismatch).

- Sparse search:
  - Uses term-weighted representations (e.g., BM25 or SPLADE). Here, sparse vectors come from `fastembed` (SPLADE‑style) when hybrid is enabled.
  - Strengths: exact term match, great for rare keywords and symbols.
  - Limitations: weaker semantic understanding; sensitive to wording.

- Hybrid search:
  - Combines both dense and sparse results on the same collection in Qdrant.
  - Blend controlled via `alpha` (0.0 = purely sparse, 1.0 = purely dense).
  - Typically improves recall and precision across a wide range of queries.

## Architecture Overview
- Embeddings
  - Dense: `HuggingFaceEmbedding(\"BAAI/bge-large-en\")` configured once in LlamaIndex `Settings`.
  - Sparse: `fastembed` SPLADE is used automatically by LlamaIndex Qdrant adapter when `enable_hybrid=True`.

- Qdrant Collections
  - `hacker-news` (dense only): stores dense vectors for vector search.
  - `hacker-news-hybrid` (dense + sparse): stores dense vectors and enables sparse vectors; required for hybrid search.

- API (`api/server.py`)
  - `GET /health`: health check.
  - `POST /collections/init`: creates/verifies both collections with proper configs.
    - Dense: `VectorParams(size=1024, distance=Cosine)` + HNSW tuning.
    - Hybrid: same dense config + `SparseVectorParams()` (turns on sparse).
  - `GET /search`: retrieval endpoint.
    - Params: `query`, `mode=vector|hybrid`, `top_k`, optional `alpha` (for hybrid).
    - Returns the top node texts; no LLM answer generation (so no hallucinations).

- Notebooks
  - `01-semantic-search-rag.ipynb`: baseline dense retrieval and RAG knobs (filters, response modes, postprocessors).
  - `02-tweaking-semantic-search.ipynb`: Qdrant tuning (HNSW, on-disk, quantization) for quality/latency/memory.
  - `03-hybrid-search.ipynb`: compares sparse-only and hybrid; shows how `alpha`, `sparse_top_k`, `similarity_top_k` affect results.

## Setup & Run
1) Choose your Qdrant
- Local via Docker Desktop
  - Start Docker Desktop (ensure `docker ps` works).
  - Run Qdrant:
    ```zsh
    docker run -p 6333:6333 qdrant/qdrant:latest
    export QDRANT_URL="http://localhost:6333"
    ```
- Or Qdrant Cloud
  - Create a cluster; set:
    ```zsh
    export QDRANT_URL="https://YOUR-CLUSTER-URL"
    export QDRANT_API_KEY="YOUR_API_KEY"
    ```

2) Start the API
```zsh
cd api
python server.py
# If 8000 is busy: PORT=8010 python server.py
```

3) Initialize collections
```zsh
curl -X POST "http://0.0.0.0:8000/collections/init?dense_collection=hacker-news&hybrid_collection=hacker-news-hybrid&dim=1024&distance=Cosine"
```

4) Query examples
- Vector search (dense only; no extra deps):
```zsh
curl "http://0.0.0.0:8000/search?query=What%20is%20the%20best%20way%20to%20learn%20programming%3F&mode=vector&top_k=5"
```
- Hybrid search (install `fastembed` first):
```zsh
pip install fastembed
curl "http://0.0.0.0:8000/search?query=What%20is%20the%20best%20way%20to%20learn%20programming%3F&mode=hybrid&top_k=5&alpha=0.3"
```

## How Retrieval Works Here
- Vector mode:
  1) The query is embedded by `BAAI/bge-large-en`.
  2) Qdrant finds nearest dense vectors in `hacker-news` (cosine similarity).
  3) The API returns the top `k` node texts.

- Hybrid mode:
  1) The query is embedded (dense) and transformed into a sparse representation (SPLADE via `fastembed`).
  2) Qdrant searches both modalities in `hacker-news-hybrid`.
  3) Results are blended using `alpha` and returned as top `k` node texts.

## Troubleshooting
- Port already in use:
  ```zsh
  lsof -i :8000
  kill -9 <PID>
  PORT=8010 python server.py
  ```
- Qdrant connection / collections missing (503 or 404):
  - Ensure `QDRANT_URL` (and `QDRANT_API_KEY` if Cloud) are set.
  - Local Docker running? `docker ps` should show qdrant.
  - Run `POST /collections/init` as above.
- Hybrid complains about FastEmbed:
  ```zsh
  pip install fastembed   # or fastembed-gpu
  ```
- Large first-run downloads (normal):
  - Embedding/sparse models (ONNX + tokenizer files) are downloaded once and cached.

## What’s NOT included yet (optional next steps)
- Ingestion endpoint: upload text/files/URLs, chunk, embed (dense + sparse), upsert into Qdrant.
- Answer synthesis endpoint: call an LLM (with citations) to generate answers strictly grounded in retrieved nodes, and log the exact prompt used.

If you want these, we can add `/ingest` and `/answer` endpoints with evaluation utilities (scores, thresholds, and citation checks).
