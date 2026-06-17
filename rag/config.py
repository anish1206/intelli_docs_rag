"""
rag/config.py
=============
Single source of truth for every tunable constant used across the project.

Paths are resolved relative to the **project root** (the directory that
contains this `rag/` package), so the same values work whether you run
`python chat.py`, `python eval/evaluate_rag.py`, or a Jupyter notebook.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# rag/config.py  →  rag/  →  project root
_RAG_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = _RAG_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
COLLECTION_NAME: str   = "pdf_documents"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int             = 5
MAX_CONTEXT_CHARS: int = 4000
SCORE_THRESHOLD: float = 0.0

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
CHUNK_SIZE: int    = 1000
CHUNK_OVERLAP: int = 200

SUPPORTED_EXTENSIONS: list[str] = [
    "*.pdf", "*.ppt", "*.pptx", "*.doc", "*.docx", "*.txt"
]

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
# Gemini (default for chat / notebook)
GEMINI_MODEL: str   = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.3

# Ollama (default for eval — fully local, no rate limits)
OLLAMA_MODEL: str    = "qwen2.5:0.5b"
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE: str = """\
You are a helpful assistant that answers questions based on the provided context.
The context contains information from various documents with source citations.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the provided context
- If the answer is not in the context, say you don't have enough information
- Include source citations in your answer when relevant
- Be specific and accurate
- If multiple documents provide information, synthesize them coherently

Answer:"""
