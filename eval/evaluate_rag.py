"""
eval/evaluate_rag.py
====================
RAG Evaluation Pipeline — thin consumer of `rag/`.

Metrics
-------
  1. Retrieval Relevance   – cosine similarity between query and retrieved chunks
  2. Groundedness          – LLM judge: is the answer supported by the context?
  3. Answer Correctness    – LLM judge: does the answer match the reference?
  4. Answer Relevance      – LLM judge: does the answer address the question?

LLM backend: Ollama (fully local, zero rate limits, no API key needed).

Results are written to:
  eval/eval_results_<timestamp>.json
  eval/eval_summary_<timestamp>.csv
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root & add it to sys.path so `rag` is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent   # .../eval/
PROJECT_ROOT = SCRIPT_DIR.parent                  # project root
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import core RAG logic from the single source of truth
# ---------------------------------------------------------------------------
from rag.pipeline import EmbeddingManager, RAGRetriever, VectorStore, ask, load_env
from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
    VECTOR_STORE_DIR,
)

# ---------------------------------------------------------------------------
# Eval-specific paths & constants
# ---------------------------------------------------------------------------
DATASET_PATH = SCRIPT_DIR / "test_dataset.csv"
LOG_DIR      = SCRIPT_DIR

timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path     = LOG_DIR / f"eval_results_{timestamp}.json"
summary_path = LOG_DIR / f"eval_summary_{timestamp}.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM setup — Ollama only (fully local)
# ---------------------------------------------------------------------------

def build_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


def _call_llm_with_retry(llm, prompt: str, retries: int = 4) -> str:
    """Exponential back-off on transient / rate-limit errors."""
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("429", "quota", "rate")):
                wait = 15 * (2 ** attempt)
                logger.warning("Rate limit. Waiting %ds (retry %d/%d)…", wait, attempt + 1, retries)
                time.sleep(wait)
            else:
                logger.warning("LLM call failed: %s", exc)
                return ""
    logger.error("All retries exhausted.")
    return ""


# ---------------------------------------------------------------------------
# Score parsing helper
# ---------------------------------------------------------------------------

def parse_score(raw: str, low: float = 0.0, high: float = 1.0) -> float:
    """Extract first numeric value from an LLM response and clamp."""
    matches = re.findall(r"\d+(?:\.\d+)?", raw)
    if not matches:
        return 0.0
    val = float(matches[0])
    if val > high:
        val /= 10.0           # normalise 0-10 scale → 0-1
    return max(low, min(high, val))


# ---------------------------------------------------------------------------
# Metric 1: Retrieval Relevance (embedding-based, no LLM)
# ---------------------------------------------------------------------------

def metric_retrieval_relevance(
    query: str,
    retrieved_docs: List[Dict],
    embedder: EmbeddingManager,
) -> float:
    from sklearn.metrics.pairwise import cosine_similarity

    if not retrieved_docs:
        return 0.0
    chunk_texts = [d["content"] for d in retrieved_docs]
    q_emb = embedder.embed([query])
    c_emb = embedder.embed(chunk_texts)
    sims  = cosine_similarity(q_emb, c_emb)[0]
    return float(np.mean(sorted(sims, reverse=True)[:3]))
    # returns the mean of he top 3 high rianked chunks


# ---------------------------------------------------------------------------
# Metrics 2-4: Combined LLM judge (single call per question)
# ---------------------------------------------------------------------------

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

groundedness – every claim in the answer is supported by the retrieved context
                (1.0 = fully supported, 0.0 = unsupported / contradicts context)

answer_correctness – Evaluate factual agreement and semantic equivalence with the reference answer. Different wording should not reduce score.
                    Score 1.0 if both answers convey the same information, even if phrased differently, else 0.0
                    If the generated answer states that it lacks information, and the reference answer contains specific facts, answer_correctness must be 0.0.

answer_relevance  – if the generated answer directly addresses the question
                    (1.0 = fully on-topic else 0.0 when off-topic)

Respond with exactly this JSON and nothing else:
{{"groundedness": <float>, "answer_correctness": <float>, "answer_relevance": <float>}}
"""


def metric_llm_judges(
    question: str,
    reference: str,
    context: str,
    answer: str,
    llm,
) -> Dict[str, float]:
    """Single LLM call → three judge scores."""
    prompt = COMBINED_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        context=context[:6000],
        answer=answer,
    )
    raw = _call_llm_with_retry(llm, prompt)

    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        scores  = json.loads(cleaned)
        return {
            "groundedness":      parse_score(str(scores.get("groundedness", 0))),
            "answer_correctness": parse_score(str(scores.get("answer_correctness", 0))),
            "answer_relevance":  parse_score(str(scores.get("answer_relevance", 0))),
        }
    except Exception:
        logger.warning("JSON parse failed for judge response: %r. Falling back to regex.", raw)

        def _extract(field: str) -> float:
            m = re.search(rf'"{field}"\s*:\s*(\d+(?:\.\d+)?)', raw)
            return parse_score(m.group(1)) if m else 0.0

        return {
            "groundedness":      _extract("groundedness"),
            "answer_correctness": _extract("answer_correctness"),
            "answer_relevance":  _extract("answer_relevance"),
        }


# ---------------------------------------------------------------------------
# RAG answer generator — delegates to rag.pipeline.ask()
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    retriever: RAGRetriever,
    llm,
) -> tuple[str, str, List[Dict]]:
    """Returns (answer, formatted_context, raw_docs)."""
    docs    = retriever.retrieve(question, top_k=TOP_K)
    context = retriever.format_context(docs)
    answer  = ask(question, retriever, llm, top_k=TOP_K)
    return answer, context, docs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation():
    import pandas as pd

    logger.info("=" * 60)
    logger.info("RAG Evaluation Pipeline")
    logger.info("=" * 60)

    # Load .env (picks up GEMINI_API_KEY etc. if present)
    load_env()

    df = pd.read_csv(DATASET_PATH)
    logger.info("Loaded %d test cases from %s", len(df), DATASET_PATH)

    # Init core components (from rag/)
    embedder  = EmbeddingManager()
    store     = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )
    print("\n Total chunks in the db:")
    print(store.collection.count())
    print()
    
    retriever = RAGRetriever(store, embedder)
    llm       = build_llm()

    logger.info("LLM: %s @ %s (Ollama — RAG + judge)", OLLAMA_MODEL, OLLAMA_BASE_URL)

    results: List[Dict[str, Any]] = []
    total = len(df)

    for idx, row in df.iterrows():
        question  = str(row["Question"]).strip()
        reference = str(row["Reference Answer"]).strip()

        logger.info("\n[%d/%d] Q: %s…", idx + 1, total, question[:80])

        t0 = time.time()
        answer, context, docs = generate_answer(question, retriever, llm)
        latency = round(time.time() - t0, 3)

        logger.info("  → Answer (%ss): %s…", latency, answer[:100])

        ret_rel = metric_retrieval_relevance(question, docs, embedder)
        logger.info("  Retrieval Relevance : %.4f", ret_rel)

        judge_scores = metric_llm_judges(question, reference, context, answer, llm)
        grounded     = judge_scores["groundedness"]
        correctness  = judge_scores["answer_correctness"]
        relevance    = judge_scores["answer_relevance"]
        logger.info("  Groundedness        : %.4f", grounded)
        logger.info("  Answer Correctness  : %.4f", correctness)
        logger.info("  Answer Relevance    : %.4f", relevance)

        results.append(
            {
                "index":           int(idx),
                "question":        question,
                "reference_answer": reference,
                "generated_answer": answer,
                "retrieved_chunks": [
                    {"content": d["content"][:300], "similarity": d["similarity"]}
                    for d in docs
                ],
                "metrics": {
                    "retrieval_relevance": round(ret_rel, 4),
                    "groundedness":        round(grounded, 4),
                    "answer_correctness":  round(correctness, 4),
                    "answer_relevance":    round(relevance, 4),
                },
                "latency_seconds": latency,
            }
        )

        time.sleep(0.2)   # tiny pause for log readability

    # -----------------------------------------------------------------------
    # Aggregate & save
    # -----------------------------------------------------------------------
    all_metrics = [r["metrics"] for r in results]

    def _agg(key: str) -> Dict[str, float]:
        vals = [m[key] for m in all_metrics]
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
            "min":  round(float(np.min(vals)), 4),
            "max":  round(float(np.max(vals)), 4),
        }

    summary = {
        "eval_timestamp":    timestamp,
        "embedding_model":   EMBEDDING_MODEL,
        "llm":               OLLAMA_MODEL,
        "num_test_cases":    total,
        "top_k_retrieval":   TOP_K,
        "aggregate": {
            "retrieval_relevance": _agg("retrieval_relevance"),
            "groundedness":        _agg("groundedness"),
            "answer_correctness":  _agg("answer_correctness"),
            "answer_relevance":    _agg("answer_relevance"),
        },
        "per_question_results": results,
    }

    log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("\nFull results saved → %s", log_path)

    csv_rows = [
        {
            "index":              r["index"],
            "question":           r["question"],
            "reference_answer":   r["reference_answer"],
            "generated_answer":   r["generated_answer"],
            "retrieval_relevance": r["metrics"]["retrieval_relevance"],
            "groundedness":        r["metrics"]["groundedness"],
            "answer_correctness":  r["metrics"]["answer_correctness"],
            "answer_relevance":    r["metrics"]["answer_relevance"],
            "latency_seconds":     r["latency_seconds"],
        }
        for r in results
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info("Summary CSV saved     → %s", summary_path)

    # Print table
    agg = summary["aggregate"]
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("%-28s %7s  %7s  %7s  %7s", "Metric", "Mean", "Std", "Min", "Max")
    logger.info("-" * 60)
    for mname in ["retrieval_relevance", "groundedness", "answer_correctness", "answer_relevance"]:
        m = agg[mname]
        logger.info("%-28s %7.4f  %7.4f  %7.4f  %7.4f", mname, m["mean"], m["std"], m["min"], m["max"])
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_evaluation()
