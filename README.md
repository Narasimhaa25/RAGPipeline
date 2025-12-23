# RAG API (Modular) — Quick Start

Production-ready, modular Retrieval-Augmented Generation (RAG) API built with FastAPI, LlamaIndex, and Qdrant. The code you should use lives in the RAG folder. Notebooks are intentionally out-of-scope for this README.

— Python 3.11 required —


## What’s Inside

```
workshop-rag-optimization-main/
├── main.py                  # Entry point (starts FastAPI)
├── RAG/                     # RAG package (use this)
│   ├── core/
│   │   └── config.py        # Qdrant client, embeddings, tracing
│   └── routes/
│       ├── collections.py   # POST /collections/init
│       ├── search.py        # GET /search
│       └── answer.py        # POST /answer
└── requirements.txt
```


## Requirements

- Python 3.11 (recommended: use `python3.11` explicitly)
- Docker (for local Qdrant) or Qdrant Cloud
- OpenAI API key (for answer generation)


## Setup

### Option A — venv (Python 3.11)

```bash
cd /path/to/workshop-rag-optimization-main
python3.11 -m venv venv
source venv/bin/activate            # macOS/Linux
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Option B — uv (fast alternative)

```bash
cd /path/to/workshop-rag-optimization-main
uv venv --python 3.11
source .venv/bin/activate           # macOS/Linux
uv pip install -r requirements.txt
```


## Configure Environment

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your-key-if-using-cloud

OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-3.5-turbo

HOST=0.0.0.0
PORT=8000
RELOAD=false
EOF
```


## Run Qdrant (Docker)

Pick one of the following:

```bash
# Pull image (first time)
docker pull qdrant/qdrant

# Simple (ephemeral)
docker run -p 6333:6333 qdrant/qdrant

# Persistent storage
docker run -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Health check
curl http://localhost:6333/health   # -> {"status":"ok"}
```


## Run the API

```bash
# From project root, with venv activated
python main.py

# Or via uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Swagger UI
# http://localhost:8000/docs
```


## Initialize Collections (once)

```bash
curl -X POST http://localhost:8000/collections/init
```


## Endpoints

- Health: `GET /health`
- Init Collections: `POST /collections/init`
- Search: `GET /search`
  - Query params: `query` (required), `mode` (vector|hybrid), `top_k` (1–25), `alpha` (0–1), `include_scores`, `include_metadata`
- Answer: `POST /answer`

Examples:

```bash
# Search (vector)
curl "http://localhost:8000/search?query=machine%20learning&mode=vector&top_k=5"

# Search (hybrid)
curl "http://localhost:8000/search?query=machine%20learning&mode=hybrid&top_k=5&alpha=0.5"

# Answer
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is RAG?",
        "mode": "vector",
        "top_k": 5,
        "include_sources": true
      }'
```


## Notes

- Code lives in the RAG package. Notebooks are intentionally excluded here.
- If hybrid search complains about FastEmbed, install it: `pip install fastembed`.
- A setuptools pin is included to suppress a known `pkg_resources` deprecation warning.


## Where to Extend

- App entry: `main.py`
- Shared config: `RAG/core/config.py`
- Routes: `RAG/routes/{collections,search,answer}.py`
# RAG Search API - Setup & Installation Guide

A modular RAG (Retrieval-Augmented Generation) API built with FastAPI, LLamaIndex, and Qdrant for semantic search and answer generation.
