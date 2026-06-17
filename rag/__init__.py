"""
rag/__init__.py
===============
Public surface of the `rag` package.

Import anything you need from here:

    from rag import EmbeddingManager, VectorStore, RAGRetriever, ask
    from rag import RAGConfig          # convenient namespace for config values

Or import from sub-modules directly for finer control:

    from rag.pipeline import EmbeddingManager
    from rag.config   import TOP_K, GEMINI_MODEL
"""

from rag.pipeline import (  # noqa: F401
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    ask,
    load_env,
)
from rag import config as RAGConfig  # noqa: F401

__all__ = [
    "EmbeddingManager",
    "VectorStore",
    "RAGRetriever",
    "ask",
    "load_env",
    "RAGConfig",
]
