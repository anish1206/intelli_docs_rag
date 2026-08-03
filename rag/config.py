"""
rag/config.py
=============
Single source of truth for every tunable constant used across the project.

Paths are resolved relative to the **project root** (the directory that
contains this `rag/` package), so the same values work whether you run
`python chat.py`, `python eval/evaluate_rag.py`, or a Jupyter notebook.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# rag/config.py  →  rag/  →  project root
_RAG_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = _RAG_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"

# Load environment variables from .env file at the project root
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
COLLECTION_NAME: str   = os.getenv("COLLECTION_NAME", "pdf_documents")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int             = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.25"))

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

SUPPORTED_EXTENSIONS: list[str] = [
    "*.pdf", "*.ppt", "*.pptx", "*.doc", "*.docx", "*.txt"
]

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
# Gemini (default for chat / notebook)
GEMINI_MODEL: str         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

# Ollama (default for eval — running locally or remotely via Ngrok/Colab)
OLLAMA_MODEL: str         = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL: str      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

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