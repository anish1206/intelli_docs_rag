"""
rag/pipeline.py

Public surface
--------------
  EmbeddingManager  – loads SentenceTransformer, generates embeddings
  VectorStore       – ChromaDB wrapper; add_documents() for ingestion
  RAGRetriever      – query the store, re-rank, format context for LLM
  ask()             – one-call RAG: retrieve → format → generate → return
  load_env()        – lightweight .env loader (no python-dotenv dependency)

Change any of these once and every consumer (chat.py, eval/, notebook)
picks it up automatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    PROJECT_ROOT,
    RAG_PROMPT_TEMPLATE,
    SCORE_THRESHOLD,
    TOP_K,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env(path: Path = PROJECT_ROOT / ".env") -> None:
    """
    Minimal .env loader — no third-party dependency required.
    Reads KEY=VALUE lines and calls os.environ.setdefault() for each.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """
    Loads a SentenceTransformer model and generates dense embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to ``config.EMBEDDING_MODEL``.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info(
            "Embedding model ready. Dimension: %d",
            self.model.get_sentence_embedding_dimension(),
        )

    # Primary API — used by RAGRetriever and eval metrics
    def embed(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Return a (len(texts), dim) float32 ndarray."""
        if not self.model:
            raise RuntimeError("Embedding model is not loaded.")
        return self.model.encode(texts, show_progress_bar=show_progress_bar)

    # Alias kept for notebook compatibility
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Alias for :meth:`embed` with a progress bar (notebook-friendly)."""
        return self.embed(texts, show_progress_bar=True)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB-backed persistent vector store.

    Parameters
    ----------
    collection_name : str
        Name of the ChromaDB collection.
    persist_directory : str | Path
        Directory where ChromaDB persists data.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path = VECTOR_STORE_DIR,
    ) -> None:
        import chromadb  # lazy import

        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)

        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Document embeddings for RAG",
                "hnsw:space" : "cosine",
            },
        )
        logger.info(
            "VectorStore ready. Collection '%s' — %d doc(s).",
            collection_name,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        """
        Add LangChain ``Document`` objects and their embeddings to the store.

        Parameters
        ----------
        documents  : list of LangChain Document objects
        embeddings : ndarray of shape (N, dim) returned by EmbeddingManager
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        ids, metadatas, texts, emb_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            texts.append(doc.page_content)
            emb_list.append(embedding.tolist())

            metadata: Dict[str, Any] = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
                elif key == "origin" and isinstance(value, dict):
                    metadata["filename"] = value.get("filename", "")
                elif key == "headings" and isinstance(value, list):
                    metadata["headings"] = ", ".join(value) if value else ""
                elif key == "dl_meta" and isinstance(value, dict):
                    doc_items = value.get("doc_items", [])
                    if doc_items:
                        prov = doc_items[0].get("prov", [])
                        if prov:
                            metadata["page_no"] = prov[0].get("page_no", 0)
                    metadata["dl_meta"] = json.dumps(value)
                else:
                    metadata[key] = json.dumps(value)

            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=emb_list,
            metadatas=metadatas,
            documents=texts,
        )
        logger.info(
            "Added %d documents. Total in collection: %d.",
            len(documents),
            self.collection.count(),
        )


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Query the vector store, re-rank results, and format context for the LLM.

    Parameters
    ----------
    vector_store      : VectorStore instance
    embedding_manager : EmbeddingManager instance
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for *query*.

        Returns a list of dicts with keys:
            id, content, metadata, similarity_score, enhanced_score, distance, rank
        """
        query_embedding = self.embedding_manager.embed([query])[0]

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k * 2,  # over-fetch for re-ranking
        }
        if filter_metadata:
            query_params["where"] = filter_metadata

        try:
            results = self.vector_store.collection.query(**query_params)
            print("\nCOLLECTION METADATA:")
            print(self.vector_store.collection.metadata)    
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            return []

        return self._process_and_rank(results, query, top_k, score_threshold)

    def _process_and_rank(
        self,
        results: Dict,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        if not results["documents"] or not results["documents"][0]:
            return []

        docs_raw  = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]
        ids       = results["ids"][0]
        
        print("\n" + "="*80)
        print("QUERY:", query)
        print("="*80)

        for i, (text, dist) in enumerate(zip(docs_raw, distances)):
            print(f"\nRANK {i+1}")
            print("DISTANCE:", dist)
            print(text[:300])

        seen: set = set()
        ranked: List[Dict[str, Any]] = []

        for i, (doc_id, text, meta, dist) in enumerate(
            zip(ids, docs_raw, metas, distances)
        ):

            # Chorma already returns ranked results, so lower the distance, better match 
            sim = 1.0 - float(dist)

            h = hash(text)
            if h in seen:
                continue
            seen.add(h)

            enhanced_score = self._enhanced_score(
                document=text,
                query=query,
                similarity_score=sim,
                metadata=meta,
            )

            if enhanced_score < score_threshold:
                continue

            ranked.append(
                {
                    "id":               doc_id,
                    "content":          text,
                    "metadata":         meta,
                    "similarity_score": sim,
                    "enhanced_score":   enhanced_score,
                    "distance":         dist,
                    "rank":             i + 1,
                    # eval-compat alias
                    "similarity":       sim,
                }
            )

        ranked.sort(key=lambda x: x["enhanced_score"], reverse=True)

        return ranked[:top_k]

        

    def _enhanced_score(
        self,
        document: str,
        query: str,
        similarity_score: float,
        metadata: Dict,
    ) -> float:
        """Boost score using heading overlap and content length signals."""
        score = similarity_score

        headings = metadata.get("headings", "")
        if headings:
            q_terms = set(query.lower().split())
            h_terms = set(headings.lower().split())
            overlap = len(q_terms & h_terms)
            if overlap > 0:
                score += 0.1 * overlap

        length = len(document)
        if length > 1000:
            score += 0.1
        elif length > 500:
            score += 0.05

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context_for_llm(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_context_length: int = MAX_CONTEXT_CHARS,
    ) -> str:
        """
        Render retrieved chunks into a rich context string for the LLM prompt.
        Includes source citations and section headings where available.
        """
        if not retrieved_docs:
            return "No relevant context found."

        parts: List[str] = []
        used = 0

        for doc in retrieved_docs:
            meta     = doc["metadata"]
            filename = meta.get("filename") or meta.get("source_file", "Unknown")
            page_no  = meta.get("page_no", "N/A")
            headings = meta.get("headings", "")

            header = f"[Source: {filename}"
            if page_no != "N/A":
                header += f", Page {page_no}"
            if headings:
                header += f", Section: {headings}"
            header += "]"

            block = f"{header}\n{doc['content']}"
            if used + len(block) > max_context_length:
                break
            parts.append(block)
            used += len(block)

        context = "\n\n---\n\n".join(parts)
        return f"Context from {len(parts)} document(s):\n\n{context}"

    # Compact alias used by the eval pipeline
    def format_context(
        self,
        docs: List[Dict],
        max_chars: int = MAX_CONTEXT_CHARS,
    ) -> str:
        """Alias for :meth:`format_context_for_llm` (eval-compat)."""
        return self.format_context_for_llm(docs, max_context_length=max_chars)


# ---------------------------------------------------------------------------
# ask() — one-call RAG entry point
# ---------------------------------------------------------------------------

def ask(
    query: str,
    retriever: RAGRetriever,
    llm: Any,
    top_k: int = TOP_K,
    max_context_length: int = MAX_CONTEXT_CHARS,
    filter_metadata: Optional[Dict] = None,
) -> str:
    """
    High-level RAG call: retrieve → format context → generate answer.

    Parameters
    ----------
    query              : The user's question.
    retriever          : A configured RAGRetriever instance.
    llm                : Any LangChain chat model (Gemini, Ollama, etc.).
    top_k              : Number of chunks to retrieve.
    max_context_length : Maximum characters of context to pass to the LLM.
    filter_metadata    : Optional ChromaDB metadata filter dict.

    Returns
    -------
    str  The LLM's answer string.
    """
    docs = retriever.retrieve(
        query,
        top_k=top_k,
        filter_metadata=filter_metadata,
    )
    if not docs:
        return "I don't have enough relevant information to answer this question."

    context = retriever.format_context_for_llm(docs, max_context_length)
    prompt  = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content
