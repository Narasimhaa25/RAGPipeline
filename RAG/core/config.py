"""
Shared configuration and initialization for the RAG API.
Located in RAG/core/config.py

Handles:
- Environment setup
- Qdrant client initialization
- Embeddings configuration
- Service context setup
- LangSmith/Tracing integration
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index import ServiceContext, set_global_service_context
import openai as openai_mod

# Load environment variables
load_dotenv()

# -------------------------
# LangSmith / Tracing Setup
# -------------------------

try:
    from langsmith.run_helpers import traceable, wrap_openai
    _LANGSMITH_HELPERS_AVAILABLE = True
except Exception:
    traceable = None
    wrap_openai = None
    _LANGSMITH_HELPERS_AVAILABLE = False

try:
    from langchain.callbacks.tracers import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    _LANGCHAIN_TRACER_AVAILABLE = True
except Exception:
    LangChainTracer = None
    CallbackManager = None
    _LANGCHAIN_TRACER_AVAILABLE = False

try:
    from llama_index.callbacks import CallbackManager as LI_CallbackManager
    _LI_CB_AVAILABLE = True
except Exception:
    LI_CallbackManager = None
    _LI_CB_AVAILABLE = False

_LI_LANGCHAIN_HANDLER_AVAILABLE = False
LI_LangChainCallbackHandler = None
for _path in (
    "llama_index.callbacks.langchain_callback",
    "llama_index.callbacks.langchain",
    "llama_index.core.callbacks.langchain_callback",
):
    try:
        mod = __import__(_path, fromlist=["LangChainCallbackHandler"])
        LI_LangChainCallbackHandler = getattr(mod, "LangChainCallbackHandler", None)
        if LI_LangChainCallbackHandler is not None:
            _LI_LANGCHAIN_HANDLER_AVAILABLE = True
            break
    except Exception:
        continue

# Check if tracing is enabled
_TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")

# Build LangChain tracer (for LangSmith)
lc_tracer = None
if _TRACING_ENABLED and _LANGCHAIN_TRACER_AVAILABLE:
    try:
        lc_tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT"),
        )
    except Exception:
        lc_tracer = None

if _TRACING_ENABLED and _LANGSMITH_HELPERS_AVAILABLE and wrap_openai is not None:
    try:
        wrap_openai(openai_mod)
    except Exception:
        pass

# Bridge: create a LlamaIndex CallbackManager that forwards events to LangChain tracer
li_callback_manager = None
if _TRACING_ENABLED and _LI_CB_AVAILABLE and _LI_LANGCHAIN_HANDLER_AVAILABLE and lc_tracer is not None:
    handler = None
    try:
        handler = LI_LangChainCallbackHandler(lc_tracer)
    except TypeError:
        try:
            handler = LI_LangChainCallbackHandler()
        except Exception:
            handler = None
    if handler is not None:
        try:
            li_callback_manager = LI_CallbackManager([handler])
        except Exception:
            li_callback_manager = None


# -------------------------
# Service Context & Embeddings
# -------------------------

# Configure embeddings once (LlamaIndex 0.9.x)
_service_context = ServiceContext.from_defaults(
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-large-en"),
    callback_manager=li_callback_manager,
)
set_global_service_context(_service_context)

# -------------------------
# Qdrant Client
# -------------------------

qdrant_client = QdrantClient(
    os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)

service_context = _service_context


# -------------------------
# Tracing Decorator
# -------------------------

def trace_decorator(name: str):
    """
    Decorate a function to emit a LangSmith trace when tracing is enabled.
    Uses langsmith.run_helpers.traceable if available; otherwise acts as a no-op.
    
    Args:
        name: Name of the traced operation
        
    Returns:
        Decorator function
    """
    tracing_enabled = _TRACING_ENABLED
    if tracing_enabled and _LANGSMITH_HELPERS_AVAILABLE and traceable is not None:
        def _decorator(fn):
            return traceable(name=name)(fn)
        return _decorator
    # Fallback: no-op decorator
    def _decorator(fn):
        return fn
    return _decorator
