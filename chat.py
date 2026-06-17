"""
chat.py
=======
Interactive CLI chat interface — thin consumer of `rag/`.

Usage
-----
    python chat.py                       # uses Gemini (default)
    LLM_BACKEND=ollama python chat.py    # use local Ollama instead

Environment variables
---------------------
  GEMINI_API_KEY   – required when LLM_BACKEND=gemini (default)
  LLM_BACKEND      – "gemini" (default) or "ollama"
  TOP_K            – override number of retrieved chunks (default: config.TOP_K)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (handles running from any directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load .env before anything that might need API keys
# ---------------------------------------------------------------------------
from rag.pipeline import EmbeddingManager, RAGRetriever, VectorStore, ask, load_env
from rag.config import (
    COLLECTION_NAME,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    TOP_K,
    VECTOR_STORE_DIR,
)

load_env()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # keep the chat REPL clean
    format="%(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env or set the env var."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        api_key=api_key,
    )


def _build_ollama():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
    )


def build_llm():
    backend = os.getenv("LLM_BACKEND", "gemini").lower()
    if backend == "ollama":
        print(f"[chat] Using Ollama backend: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        return _build_ollama()
    print(f"[chat] Using Gemini backend: {GEMINI_MODEL}")
    return _build_gemini()


# ---------------------------------------------------------------------------
# Bootstrap RAG pipeline (once)
# ---------------------------------------------------------------------------

def bootstrap():
    """Initialise all heavy components. Called once at startup."""
    print("[chat] Initialising RAG pipeline…")
    embedder  = EmbeddingManager()
    store     = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )
    retriever = RAGRetriever(store, embedder)
    llm       = build_llm()
    print("[chat] Ready. Type your question (or 'exit' / 'quit' to leave).\n")
    return retriever, llm


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def chat_loop(retriever: RAGRetriever, llm, top_k: int = TOP_K) -> None:
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] Goodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("[chat] Goodbye!")
            break

        answer = ask(query, retriever, llm, top_k=top_k)
        print(f"\nAssistant: {answer}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    top_k = int(os.getenv("TOP_K", str(TOP_K)))
    retriever, llm = bootstrap()
    chat_loop(retriever, llm, top_k=top_k)


if __name__ == "__main__":
    main()
