"""
RAG Evaluation Pipeline for 3_docling.ipynb
============================================
Metrics:
  1. Retrieval Relevance   – cosine similarity between query and retrieved chunks
  2. Groundedness          – LLM judge: is the answer supported by the retrieved context?
  3. Answer Correctness    – LLM judge: does the answer match the reference answer?
  4. Answer Relevance      – LLM judge: does the answer actually address the question?

LLM backend : Ollama qwen2.5:0.5b — fully local, zero rate limits, no API key needed.

Results are written to eval/eval_results_<timestamp>.json and a summary CSV.
"""

import os
import sys
import json
import time
import logging
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Resolve paths relative to project root (works whether run from repo root
# or from the eval/ directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../eval/
PROJECT_ROOT = SCRIPT_DIR.parent                       # project root
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DATASET_PATH = SCRIPT_DIR / "test_dataset.csv"
LOG_DIR = SCRIPT_DIR


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"eval_results_{timestamp}.json"
summary_path = LOG_DIR / f"eval_summary_{timestamp}.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def load_env(path: Path = PROJECT_ROOT / ".env"):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
# GEMINI_API_KEY is no longer required — everything runs through Ollama


# ---------------------------------------------------------------------------
# Imports (after env load so API keys are available)
# ---------------------------------------------------------------------------
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Rebuild the retriever from the persisted vector store
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_documents"
TOP_K = 5
MAX_CONTEXT_CHARS = 4000


class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model ready.")

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class VectorStoreRetriever:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: EmbeddingManager):
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_collection(name=collection_name)
        logger.info(
            f"Connected to collection '{collection_name}' "
            f"({self.collection.count()} docs)"
        )

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed([query])[0]
        results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
        )
        docs = []
        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(
                {
                    "content": doc_text,
                    "metadata": meta,
                    "similarity": float(1 - dist),
                }
            )
        return docs

    def format_context(self, docs: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        parts = []
        used = 0
        for d in docs:
            fname = d["metadata"].get("filename") or d["metadata"].get("source_file", "unknown")
            page = d["metadata"].get("page_no", "?")
            header = f"[Source: {fname}, Page {page}]"
            block = f"{header}\n{d['content']}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n---\n\n".join(parts) if parts else "No context retrieved."


# ---------------------------------------------------------------------------
# 2.  LLM judge helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2.  LLM setup — Ollama only, fully local, no rate limits
# ---------------------------------------------------------------------------
OLLAMA_MODEL    = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"


def build_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


def _call_llm_with_retry(llm, prompt: str, retries: int = 4) -> str:
    """Call an LLM with exponential back-off on rate-limit / transient errors."""
    for attempt in range(retries):
        try:
            resp = llm.invoke(prompt)
            return resp.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err:
                wait = 15 * (2 ** attempt)
                logger.warning(f"Rate limit hit. Waiting {wait}s (retry {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                logger.warning(f"LLM call failed: {e}")
                return ""
    logger.error("All retries exhausted.")
    return ""


def parse_score(raw: str, low: float = 0.0, high: float = 1.0) -> float:
    """Extract the first float/int from an LLM response and clamp it."""
    import re
    matches = re.findall(r"\d+(?:\.\d+)?", raw)
    if not matches:
        return 0.0
    val = float(matches[0])
    # If LLM returned a 0-10 scale, normalise
    if val > high:
        val = val / 10.0
    return max(low, min(high, val))


# ---------------------------------------------------------------------------
# 3.  Metrics
# ---------------------------------------------------------------------------

def metric_retrieval_relevance(query: str, retrieved_docs: List[Dict], embedder: EmbeddingManager) -> float:
    """
    Average cosine similarity between the query embedding and each
    retrieved chunk embedding. No LLM call needed.
    """
    if not retrieved_docs:
        return 0.0
    chunk_texts = [d["content"] for d in retrieved_docs]
    q_emb = embedder.embed([query])
    c_emb = embedder.embed(chunk_texts)
    sims = cosine_similarity(q_emb, c_emb)[0]
    return float(np.mean(sims))


# Single combined prompt → 1 LLM call covers groundedness + correctness + relevance
COMBINED_JUDGE_PROMPT = """\
You are a strict RAG evaluation judge. Evaluate the GENERATED ANSWER on three \
dimensions and respond with ONLY a valid JSON object — no markdown, no explanation.

QUESTION: {question}

REFERENCE ANSWER (ground truth):
{reference}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Score each dimension from 0.0 to 1.0:

groundedness     – every claim in the answer is supported by the retrieved context
                   (1.0 = fully supported, 0.0 = unsupported / contradicts context)

answer_correctness – the generated answer matches the reference answer factually
                   (1.0 = semantically equivalent, 0.0 = wrong / unrelated)

answer_relevance – the generated answer directly addresses the question
                   (1.0 = fully on-topic, 0.0 = completely off-topic)

Respond with exactly this JSON and nothing else:
{{"groundedness": <float>, "answer_correctness": <float>, "answer_relevance": <float>}}
"""


def metric_llm_judges(question: str, reference: str, context: str, answer: str, llm) -> Dict[str, float]:
    """Single LLM call that returns all three judge scores."""
    import json as _json
    import re as _re

    prompt = COMBINED_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        context=context[:3000],   # cap context to keep prompt manageable
        answer=answer,
    )
    raw = _call_llm_with_retry(llm, prompt)

    # Try to parse JSON; fall back to regex extraction per field
    try:
        # Strip potential markdown code fences
        cleaned = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        scores = _json.loads(cleaned)
        return {
            "groundedness":      parse_score(str(scores.get("groundedness", 0))),
            "answer_correctness": parse_score(str(scores.get("answer_correctness", 0))),
            "answer_relevance":  parse_score(str(scores.get("answer_relevance", 0))),
        }
    except Exception:
        logger.warning(f"JSON parse failed for judge response: {raw!r}. Falling back to regex.")
        def _extract(field: str) -> float:
            m = _re.search(rf'"{field}"\s*:\s*(\d+(?:\.\d+)?)', raw)
            return parse_score(m.group(1)) if m else 0.0
        return {
            "groundedness":      _extract("groundedness"),
            "answer_correctness": _extract("answer_correctness"),
            "answer_relevance":  _extract("answer_relevance"),
        }


# ---------------------------------------------------------------------------
# 4.  RAG answer generator (mirrors ask_gemini from 3_docling.ipynb)
# ---------------------------------------------------------------------------
RAG_PROMPT = """\
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

Answer:"""


def generate_answer(question: str, retriever: VectorStoreRetriever, llm) -> tuple[str, str, List[Dict]]:
    """Returns (answer, formatted_context, raw_docs). Uses Gemini (production model)."""
    docs = retriever.retrieve(question, top_k=TOP_K)
    context = retriever.format_context(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = _call_llm_with_retry(llm, prompt)
    return answer, context, docs


# ---------------------------------------------------------------------------
# 5.  Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation():
    logger.info("=" * 60)
    logger.info("RAG Evaluation Pipeline – 3_docling.ipynb")
    logger.info("=" * 60)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded {len(df)} test cases from {DATASET_PATH}")

    # Init components
    embedder  = EmbeddingManager()
    retriever = VectorStoreRetriever(VECTOR_STORE_DIR, COLLECTION_NAME, embedder)
    llm       = build_llm()

    logger.info(f"LLM : {OLLAMA_MODEL} (Ollama local — RAG + judge)")

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        question = str(row["Question"]).strip()
        reference = str(row["Reference Answer"]).strip()

        logger.info(f"\n[{idx + 1}/{total}] Q: {question[:80]}...")

        # Generate answer via RAG (Ollama)
        t0 = time.time()
        answer, context, docs = generate_answer(question, retriever, llm)
        latency = round(time.time() - t0, 3)

        logger.info(f"  → Answer ({latency}s): {answer[:100]}...")

        # --- Metric 1: Retrieval Relevance (embedding-based, no LLM) ---
        ret_rel = metric_retrieval_relevance(question, docs, embedder)
        logger.info(f"  Retrieval Relevance : {ret_rel:.4f}")

        # --- Metrics 2-4: single Ollama judge call ---
        judge_scores = metric_llm_judges(question, reference, context, answer, llm)
        grounded    = judge_scores["groundedness"]
        correctness = judge_scores["answer_correctness"]
        relevance   = judge_scores["answer_relevance"]
        logger.info(f"  Groundedness        : {grounded:.4f}")
        logger.info(f"  Answer Correctness  : {correctness:.4f}")
        logger.info(f"  Answer Relevance    : {relevance:.4f}")

        results.append(
            {
                "index": int(idx),
                "question": question,
                "reference_answer": reference,
                "generated_answer": answer,
                "retrieved_chunks": [
                    {"content": d["content"][:300], "similarity": d["similarity"]}
                    for d in docs
                ],
                "metrics": {
                    "retrieval_relevance": round(ret_rel, 4),
                    "groundedness": round(grounded, 4),
                    "answer_correctness": round(correctness, 4),
                    "answer_relevance": round(relevance, 4),
                },
                "latency_seconds": latency,
            }
        )

        # Judge runs locally — no sleep needed. Tiny pause just for readability.
        time.sleep(0.2)

    # ---------------------------------------------------------------------------
    # 6.  Aggregate & save
    # ---------------------------------------------------------------------------
    all_metrics = [r["metrics"] for r in results]
    summary = {
        "eval_timestamp": timestamp,
        "pipeline": "3_docling.ipynb",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm": OLLAMA_MODEL,
        "num_test_cases": total,
        "top_k_retrieval": TOP_K,
        "aggregate": {
            "retrieval_relevance": {
                "mean": round(float(np.mean([m["retrieval_relevance"] for m in all_metrics])), 4),
                "std": round(float(np.std([m["retrieval_relevance"] for m in all_metrics])), 4),
                "min": round(float(np.min([m["retrieval_relevance"] for m in all_metrics])), 4),
                "max": round(float(np.max([m["retrieval_relevance"] for m in all_metrics])), 4),
            },
            "groundedness": {
                "mean": round(float(np.mean([m["groundedness"] for m in all_metrics])), 4),
                "std": round(float(np.std([m["groundedness"] for m in all_metrics])), 4),
                "min": round(float(np.min([m["groundedness"] for m in all_metrics])), 4),
                "max": round(float(np.max([m["groundedness"] for m in all_metrics])), 4),
            },
            "answer_correctness": {
                "mean": round(float(np.mean([m["answer_correctness"] for m in all_metrics])), 4),
                "std": round(float(np.std([m["answer_correctness"] for m in all_metrics])), 4),
                "min": round(float(np.min([m["answer_correctness"] for m in all_metrics])), 4),
                "max": round(float(np.max([m["answer_correctness"] for m in all_metrics])), 4),
            },
            "answer_relevance": {
                "mean": round(float(np.mean([m["answer_relevance"] for m in all_metrics])), 4),
                "std": round(float(np.std([m["answer_relevance"] for m in all_metrics])), 4),
                "min": round(float(np.min([m["answer_relevance"] for m in all_metrics])), 4),
                "max": round(float(np.max([m["answer_relevance"] for m in all_metrics])), 4),
            },
        },
        "per_question_results": results,
    }

    # Save full JSON log
    log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nFull results saved → {log_path}")

    # Save summary CSV (one row per question)
    csv_rows = []
    for r in results:
        csv_rows.append(
            {
                "index": r["index"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "generated_answer": r["generated_answer"],
                "retrieval_relevance": r["metrics"]["retrieval_relevance"],
                "groundedness": r["metrics"]["groundedness"],
                "answer_correctness": r["metrics"]["answer_correctness"],
                "answer_relevance": r["metrics"]["answer_relevance"],
                "latency_seconds": r["latency_seconds"],
            }
        )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Summary CSV saved     → {summary_path}")

    # Print final summary table
    agg = summary["aggregate"]
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<28} {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    logger.info("-" * 60)
    for metric_name in ["retrieval_relevance", "groundedness", "answer_correctness", "answer_relevance"]:
        m = agg[metric_name]
        logger.info(
            f"{metric_name:<28} {m['mean']:>7.4f}  {m['std']:>7.4f}  {m['min']:>7.4f}  {m['max']:>7.4f}"
        )
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_evaluation()
