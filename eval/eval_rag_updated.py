# """
# eval/evaluate_rag.py
# ====================

# RAG Evaluation Pipeline

# Metrics
# -------
# 1. Retrieval Relevance
#    LLM judges whether each retrieved chunk contains evidence needed
#    to answer the question correctly.

#    Derived metrics:
#    - Hit@K
#    - Precision@K
#    - Mean Relevant Chunks

# 2. Groundedness
#    LLM judges whether the generated answer is supported by the
#    retrieved context.

# 3. Answer Correctness
#    LLM judges whether the generated answer agrees with the
#    reference answer.

# 4. Answer Relevance
#    LLM judges whether the generated answer directly answers
#    the question.

# Dataset
# -------
# The evaluation dataset only needs:

#     Question
#     Reference Answer

# No chunk IDs or vector database changes are required.

# Results:
#     eval/eval_results_<timestamp>.json
#     eval/eval_summary_<timestamp>.csv
# """

# from __future__ import annotations

# import csv
# import json
# import logging
# import re
# import sys
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List

# import numpy as np


# # ============================================================================
# # PROJECT PATH
# # ============================================================================

# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent

# sys.path.insert(0, str(PROJECT_ROOT))


# # ============================================================================
# # RAG IMPORTS
# # ============================================================================

# from rag.pipeline import (
#     EmbeddingManager,
#     RAGRetriever,
#     VectorStore,
#     ask,
#     load_env,
# )

# from rag.config import (
#     COLLECTION_NAME,
#     EMBEDDING_MODEL,
#     OLLAMA_BASE_URL,
#     OLLAMA_MODEL,
#     TOP_K,
#     VECTOR_STORE_DIR,
# )


# # ============================================================================
# # PATHS
# # ============================================================================

# DATASET_PATH = SCRIPT_DIR / "test_dataset.csv"
# LOG_DIR = SCRIPT_DIR

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# log_path = LOG_DIR / f"eval_results_{timestamp}.json"
# summary_path = LOG_DIR / f"eval_summary_{timestamp}.csv"


# # ============================================================================
# # LOGGING
# # ============================================================================

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ],
# )

# logger = logging.getLogger(__name__)


# # ============================================================================
# # LLM
# # ============================================================================

# def build_llm():
#     """
#     Build the local Ollama judge/generation model.
#     """

#     from langchain_ollama import ChatOllama

#     return ChatOllama(
#         model=OLLAMA_MODEL,
#         base_url=OLLAMA_BASE_URL,
#         temperature=0.0,
#     )


# def _call_llm_with_retry(
#     llm,
#     prompt: str,
#     retries: int = 3,
# ) -> str:
#     """
#     Call the LLM with retry support.
#     """

#     for attempt in range(retries):

#         try:

#             response = llm.invoke(prompt)

#             return response.content.strip()

#         except Exception as exc:

#             logger.warning(
#                 "LLM call failed on attempt %d/%d: %s",
#                 attempt + 1,
#                 retries,
#                 exc,
#             )

#             if attempt < retries - 1:
#                 time.sleep(2 ** attempt)

#     logger.error("All LLM retries failed.")

#     return ""


# # ============================================================================
# # JSON PARSING
# # ============================================================================

# def clean_json_response(raw: str) -> str:
#     """
#     Remove markdown code fences and surrounding text where possible.
#     """

#     raw = raw.strip()

#     raw = re.sub(
#         r"```(?:json)?",
#         "",
#         raw,
#         flags=re.IGNORECASE,
#     )

#     raw = raw.replace("```", "").strip()

#     start = raw.find("{")
#     end = raw.rfind("}")

#     if start != -1 and end != -1:
#         return raw[start:end + 1]

#     return raw


# def parse_json_response(raw: str) -> Dict[str, Any]:

#     cleaned = clean_json_response(raw)

#     try:

#         return json.loads(cleaned)

#     except Exception:

#         logger.warning(
#             "Could not parse JSON response: %r",
#             raw,
#         )

#         return {}


# def clamp_score(value: Any) -> float:

#     try:

#         value = float(value)

#     except Exception:

#         return 0.0

#     return max(0.0, min(1.0, value))


# # ============================================================================
# # METRIC 1
# # RETRIEVAL RELEVANCE
# # ============================================================================

# RETRIEVAL_JUDGE_PROMPT = """
# You are evaluating one retrieved chunk from a Retrieval-Augmented Generation
# (RAG) system.

# Your task is to determine whether this chunk contains information that is
# actually useful and sufficient to help answer the question correctly.

# Do NOT judge based only on topic similarity.

# A chunk is RELEVANT if:

# 1. It directly contains the answer, OR
# 2. It contains factual evidence necessary to derive the answer, OR
# 3. It contains an essential part of the answer.

# A chunk is NOT RELEVANT if:

# 1. It only discusses the same broad topic,
# 2. It is merely semantically similar,
# 3. It contains unrelated information,
# 4. It does not help answer the specific question.

# QUESTION:
# {question}

# REFERENCE ANSWER:
# {reference}

# RETRIEVED CHUNK:
# {chunk}

# Return ONLY valid JSON:

# {{
#     "relevant": 0 or 1,
#     "score": 0.0 to 1.0,
#     "reason": "short explanation"
# }}
# """


# def judge_retrieved_chunk(
#     question: str,
#     reference: str,
#     chunk: str,
#     llm,
# ) -> Dict[str, Any]:
#     """
#     Judge whether one retrieved chunk is relevant to answering the question.
#     """

#     prompt = RETRIEVAL_JUDGE_PROMPT.format(
#         question=question,
#         reference=reference,
#         chunk=chunk,
#     )

#     raw = _call_llm_with_retry(llm, prompt)

#     result = parse_json_response(raw)

#     return {
#         "relevant": int(result.get("relevant", 0)),
#         "score": clamp_score(result.get("score", 0.0)),
#         "reason": str(result.get("reason", "")),
#     }


# def metric_retrieval_relevance(
#     question: str,
#     reference: str,
#     retrieved_docs: List[Dict],
#     llm,
# ) -> Dict[str, Any]:
#     """
#     Evaluate every retrieved chunk individually.

#     Returns:

#         hit_at_k
#         precision_at_k
#         mean_relevance
#         chunk-level judgments
#     """

#     if not retrieved_docs:

#         return {
#             "hit_at_k": 0.0,
#             "precision_at_k": 0.0,
#             "mean_relevance": 0.0,
#             "chunk_judgments": [],
#         }

#     judgments = []

#     for rank, doc in enumerate(retrieved_docs, start=1):

#         judgment = judge_retrieved_chunk(
#             question=question,
#             reference=reference,
#             chunk=doc["content"],
#             llm=llm,
#         )

#         judgment["rank"] = rank
#         judgment["distance"] = doc.get("distance")
#         judgment["similarity"] = doc.get("similarity")

#         judgments.append(judgment)

#     relevant_flags = [
#         j["relevant"]
#         for j in judgments
#     ]

#     scores = [
#         j["score"]
#         for j in judgments
#     ]

#     hit_at_k = float(
#         any(flag == 1 for flag in relevant_flags)
#     )

#     precision_at_k = float(
#         np.mean(relevant_flags)
#     ) if relevant_flags else 0.0

#     mean_relevance = float(
#         np.mean(scores)
#     ) if scores else 0.0

#     return {
#         "hit_at_k": round(hit_at_k, 4),
#         "precision_at_k": round(precision_at_k, 4),
#         "mean_relevance": round(mean_relevance, 4),
#         "chunk_judgments": judgments,
#     }


# # ============================================================================
# # METRICS 2-4
# # GROUNDNESS, CORRECTNESS, RELEVANCE
# # ============================================================================

# ANSWER_JUDGE_PROMPT = """
# You are a strict evaluator for a Retrieval-Augmented Generation system.

# Evaluate the GENERATED ANSWER on exactly three independent dimensions.

# QUESTION:
# {question}

# REFERENCE ANSWER:
# {reference}

# RETRIEVED CONTEXT:
# {context}

# GENERATED ANSWER:
# {answer}


# ------------------------------------------------------------
# 1. GROUNDEDNESS
# ------------------------------------------------------------

# Determine whether the factual claims made in the generated answer are
# supported by the retrieved context.

# Important:

# - Every factual claim must be supported by the context.
# - If the answer contains unsupported factual claims, reduce the score.
# - If the answer contradicts the context, the score should be low.
# - If the answer says it does not have enough information and makes no
#   unsupported claims, groundedness can be high.
# - Do not confuse correctness with groundedness.

# ------------------------------------------------------------
# 2. ANSWER CORRECTNESS
# ------------------------------------------------------------

# Determine whether the generated answer conveys the same factual answer as
# the reference answer.

# Important:

# - The wording does not need to be identical.
# - Semantic equivalence is acceptable.
# - The answer must contain the important facts required by the reference.
# - Being on the same general topic is NOT enough.
# - A vague answer should not receive full credit.
# - An answer that says it lacks information when the reference contains the
#   answer is incorrect.
# - Missing important facts should reduce the score.
# - Contradictory facts should reduce the score.

# ------------------------------------------------------------
# 3. ANSWER RELEVANCE
# ------------------------------------------------------------

# Determine whether the generated answer directly answers the question.

# Important:

# - Being related to the topic is not enough.
# - The answer must address what was specifically asked.
# - A response that discusses surrounding concepts but does not answer the
#   actual question should receive a low score.
# - Concise answers can receive full credit if they directly answer the question.

# ------------------------------------------------------------

# Return ONLY valid JSON:

# {{
#     "groundedness": 0.0,
#     "answer_correctness": 0.0,
#     "answer_relevance": 0.0,
#     "reasoning": {{
#         "groundedness": "short explanation",
#         "answer_correctness": "short explanation",
#         "answer_relevance": "short explanation"
#     }}
# }}
# """


# def metric_llm_judges(
#     question: str,
#     reference: str,
#     context: str,
#     answer: str,
#     llm,
# ) -> Dict[str, Any]:
#     """
#     Evaluate groundedness, answer correctness, and answer relevance.
#     """

#     prompt = ANSWER_JUDGE_PROMPT.format(
#         question=question,
#         reference=reference,
#         context=context[:8000],
#         answer=answer,
#     )

#     raw = _call_llm_with_retry(
#         llm,
#         prompt,
#     )

#     result = parse_json_response(raw)

#     reasoning = result.get(
#         "reasoning",
#         {},
#     )

#     return {
#         "groundedness": clamp_score(
#             result.get("groundedness", 0.0)
#         ),

#         "answer_correctness": clamp_score(
#             result.get("answer_correctness", 0.0)
#         ),

#         "answer_relevance": clamp_score(
#             result.get("answer_relevance", 0.0)
#         ),

#         "reasoning": reasoning,
#     }


# # ============================================================================
# # GENERATE ANSWER
# # ============================================================================

# def generate_answer(
#     question: str,
#     retriever: RAGRetriever,
#     llm,
# ) -> tuple[str, str, List[Dict]]:
#     """
#     Retrieve documents, format context, and generate answer.

#     Returns:

#         answer
#         formatted_context
#         retrieved_documents
#     """

#     docs = retriever.retrieve(
#         question,
#         top_k=TOP_K,
#     )

#     context = retriever.format_context(
#         docs,
#     )

#     if not docs:

#         answer = (
#             "I don't have enough relevant information "
#             "to answer this question."
#         )

#     else:

#         answer = ask(
#             question,
#             retriever,
#             llm,
#             top_k=TOP_K,
#         )

#     return answer, context, docs


# # ============================================================================
# # MAIN EVALUATION
# # ============================================================================

# def run_evaluation():

#     import pandas as pd

#     logger.info("=" * 60)
#     logger.info("RAG Evaluation Pipeline")
#     logger.info("=" * 60)

#     load_env()

#     # ------------------------------------------------------------
#     # Load dataset
#     # ------------------------------------------------------------

#     df = pd.read_csv(
#         DATASET_PATH,
#     )

#     required_columns = {
#         "Question",
#         "Reference Answer",
#     }

#     missing_columns = required_columns - set(df.columns)

#     if missing_columns:

#         raise ValueError(
#             f"Dataset is missing required columns: {missing_columns}"
#         )

#     logger.info(
#         "Loaded %d test cases from %s",
#         len(df),
#         DATASET_PATH,
#     )

#     # ------------------------------------------------------------
#     # Initialize RAG components
#     # ------------------------------------------------------------

#     embedder = EmbeddingManager()

#     store = VectorStore(
#         collection_name=COLLECTION_NAME,
#         persist_directory=VECTOR_STORE_DIR,
#     )

#     logger.info(
#         "Total chunks in vector database: %d",
#         store.collection.count(),
#     )

#     retriever = RAGRetriever(
#         store,
#         embedder,
#     )

#     llm = build_llm()

#     logger.info(
#         "LLM: %s @ %s",
#         OLLAMA_MODEL,
#         OLLAMA_BASE_URL,
#     )

#     # ------------------------------------------------------------
#     # Evaluation loop
#     # ------------------------------------------------------------

#     results: List[Dict[str, Any]] = []

#     total = len(df)

#     for idx, row in df.iterrows():

#         question = str(
#             row["Question"]
#         ).strip()

#         reference = str(
#             row["Reference Answer"]
#         ).strip()

#         logger.info(
#             "\n[%d/%d] Q: %s…",
#             idx + 1,
#             total,
#             question[:100],
#         )

#         start_time = time.time()

#         # --------------------------------------------------------
#         # Generate answer
#         # --------------------------------------------------------

#         answer, context, docs = generate_answer(
#             question,
#             retriever,
#             llm,
#         )

#         latency = round(
#             time.time() - start_time,
#             3,
#         )

#         logger.info(
#             "  → Answer (%ss): %s…",
#             latency,
#             answer[:150],
#         )

#         # --------------------------------------------------------
#         # Retrieval evaluation
#         # --------------------------------------------------------

#         retrieval_metrics = metric_retrieval_relevance(
#             question=question,
#             reference=reference,
#             retrieved_docs=docs,
#             llm=llm,
#         )

#         logger.info(
#             "  Hit@%d              : %.4f",
#             TOP_K,
#             retrieval_metrics["hit_at_k"],
#         )

#         logger.info(
#             "  Retrieval Precision : %.4f",
#             retrieval_metrics["precision_at_k"],
#         )

#         logger.info(
#             "  Mean Chunk Relevance: %.4f",
#             retrieval_metrics["mean_relevance"],
#         )

#         # --------------------------------------------------------
#         # Answer evaluation
#         # --------------------------------------------------------

#         answer_metrics = metric_llm_judges(
#             question=question,
#             reference=reference,
#             context=context,
#             answer=answer,
#             llm=llm,
#         )

#         logger.info(
#             "  Groundedness        : %.4f",
#             answer_metrics["groundedness"],
#         )

#         logger.info(
#             "  Answer Correctness  : %.4f",
#             answer_metrics["answer_correctness"],
#         )

#         logger.info(
#             "  Answer Relevance    : %.4f",
#             answer_metrics["answer_relevance"],
#         )

#         # --------------------------------------------------------
#         # Save per-question result
#         # --------------------------------------------------------

#         results.append(
#             {
#                 "index": int(idx),

#                 "question": question,

#                 "reference_answer": reference,

#                 "generated_answer": answer,

#                 "retrieved_chunks": [
#                     {
#                         "rank": i + 1,
#                         "content": doc["content"],
#                         "distance": doc.get("distance"),
#                         "similarity": doc.get("similarity"),
#                     }

#                     for i, doc in enumerate(docs)
#                 ],

#                 "retrieval_evaluation": retrieval_metrics,

#                 "answer_evaluation": answer_metrics,

#                 "latency_seconds": latency,
#             }
#         )

#         time.sleep(0.2)

#     # =========================================================================
#     # AGGREGATION
#     # =========================================================================

#     def aggregate_metric(
#         values: List[float],
#     ) -> Dict[str, float]:

#         if not values:

#             return {
#                 "mean": 0.0,
#                 "std": 0.0,
#                 "min": 0.0,
#                 "max": 0.0,
#             }

#         return {
#             "mean": round(float(np.mean(values)), 4),
#             "std": round(float(np.std(values)), 4),
#             "min": round(float(np.min(values)), 4),
#             "max": round(float(np.max(values)), 4),
#         }

#     # ------------------------------------------------------------
#     # Collect metric values
#     # ------------------------------------------------------------

#     hit_values = [
#         r["retrieval_evaluation"]["hit_at_k"]
#         for r in results
#     ]

#     precision_values = [
#         r["retrieval_evaluation"]["precision_at_k"]
#         for r in results
#     ]

#     relevance_values = [
#         r["retrieval_evaluation"]["mean_relevance"]
#         for r in results
#     ]

#     groundedness_values = [
#         r["answer_evaluation"]["groundedness"]
#         for r in results
#     ]

#     correctness_values = [
#         r["answer_evaluation"]["answer_correctness"]
#         for r in results
#     ]

#     answer_relevance_values = [
#         r["answer_evaluation"]["answer_relevance"]
#         for r in results
#     ]

#     # =========================================================================
#     # SUMMARY
#     # =========================================================================

#     summary = {

#         "eval_timestamp": timestamp,

#         "embedding_model": EMBEDDING_MODEL,

#         "llm": OLLAMA_MODEL,

#         "num_test_cases": total,

#         "top_k_retrieval": TOP_K,

#         "aggregate": {

#             "retrieval_hit_at_k": aggregate_metric(
#                 hit_values
#             ),

#             "retrieval_precision_at_k": aggregate_metric(
#                 precision_values
#             ),

#             "retrieval_mean_chunk_relevance": aggregate_metric(
#                 relevance_values
#             ),

#             "groundedness": aggregate_metric(
#                 groundedness_values
#             ),

#             "answer_correctness": aggregate_metric(
#                 correctness_values
#             ),

#             "answer_relevance": aggregate_metric(
#                 answer_relevance_values
#             ),
#         },

#         "per_question_results": results,
#     }

#     # =========================================================================
#     # SAVE JSON
#     # =========================================================================

#     log_path.write_text(
#         json.dumps(
#             summary,
#             indent=2,
#             ensure_ascii=False,
#         ),
#         encoding="utf-8",
#     )

#     logger.info(
#         "\nFull results saved → %s",
#         log_path,
#     )

#     # =========================================================================
#     # SAVE CSV
#     # =========================================================================

#     csv_rows = []

#     for r in results:

#         csv_rows.append(
#             {

#                 "index": r["index"],

#                 "question": r["question"],

#                 "reference_answer": r["reference_answer"],

#                 "generated_answer": r["generated_answer"],

#                 "retrieval_hit_at_k": (
#                     r["retrieval_evaluation"]["hit_at_k"]
#                 ),

#                 "retrieval_precision_at_k": (
#                     r["retrieval_evaluation"]["precision_at_k"]
#                 ),

#                 "retrieval_mean_chunk_relevance": (
#                     r["retrieval_evaluation"]["mean_relevance"]
#                 ),

#                 "groundedness": (
#                     r["answer_evaluation"]["groundedness"]
#                 ),

#                 "answer_correctness": (
#                     r["answer_evaluation"]["answer_correctness"]
#                 ),

#                 "answer_relevance": (
#                     r["answer_evaluation"]["answer_relevance"]
#                 ),

#                 "latency_seconds": r["latency_seconds"],
#             }
#         )

#     if csv_rows:

#         with open(
#             summary_path,
#             "w",
#             newline="",
#             encoding="utf-8",
#         ) as f:

#             writer = csv.DictWriter(
#                 f,
#                 fieldnames=csv_rows[0].keys(),
#             )

#             writer.writeheader()

#             writer.writerows(csv_rows)

#     logger.info(
#         "Summary CSV saved → %s",
#         summary_path,
#     )

#     # =========================================================================
#     # PRINT SUMMARY
#     # =========================================================================

#     aggregate = summary["aggregate"]

#     logger.info("\n" + "=" * 70)
#     logger.info("EVALUATION SUMMARY")
#     logger.info("=" * 70)

#     logger.info(
#         "%-38s %7s  %7s  %7s  %7s",
#         "Metric",
#         "Mean",
#         "Std",
#         "Min",
#         "Max",
#     )

#     logger.info("-" * 70)

#     metric_names = [

#         "retrieval_hit_at_k",

#         "retrieval_precision_at_k",

#         "retrieval_mean_chunk_relevance",

#         "groundedness",

#         "answer_correctness",

#         "answer_relevance",
#     ]

#     for metric_name in metric_names:

#         metric = aggregate[metric_name]

#         logger.info(

#             "%-38s %7.4f  %7.4f  %7.4f  %7.4f",

#             metric_name,

#             metric["mean"],

#             metric["std"],

#             metric["min"],

#             metric["max"],
#         )

#     logger.info("=" * 70)

#     return summary


# # ============================================================================
# # ENTRY POINT
# # ============================================================================

# if __name__ == "__main__":

#     run_evaluation()



"""
eval/evaluate_rag.py
====================

RAG Evaluation Pipeline

Metrics
-------
1. Retrieval Relevance
   LLM judges whether each retrieved chunk contains evidence needed
   to answer the question correctly.

   Derived metrics:
   - Hit@K
   - Precision@K
   - Mean Relevant Chunks

2. Groundedness
   LLM judges whether the generated answer is supported by the
   retrieved context.

3. Answer Correctness
   LLM judges whether the generated answer agrees with the
   reference answer.

4. Answer Relevance
   LLM judges whether the generated answer directly answers
   the question.

Dataset
-------
The evaluation dataset only needs:

    Question
    Reference Answer

No chunk IDs or vector database changes are required.

Results:
    eval/eval_results_<timestamp>.json
    eval/eval_summary_<timestamp>.csv
"""
# ==========================================================================================
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ============================================================================
# PROJECT PATH
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# RAG IMPORTS
# ============================================================================

from rag.pipeline import (
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    ask,
    load_env,
)

from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
    VECTOR_STORE_DIR,
)


# ============================================================================
# PATHS
# ============================================================================

DATASET_PATH = SCRIPT_DIR / "test_dataset.csv"
LOG_DIR = SCRIPT_DIR

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = LOG_DIR / f"eval_results_{timestamp}.json"
summary_path = LOG_DIR / f"eval_summary_{timestamp}.csv"


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
)

logger = logging.getLogger(__name__)


# ============================================================================
# LLM
# ============================================================================

def build_llm():
    """
    Build the local or remote Ollama judge/generation model.
    Prioritizes OLLAMA_BASE_URL from environment/.env.
    """
    from langchain_ollama import ChatOllama

    # Ensures .env is loaded before grabbing the environment variable
    load_env()
    target_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=target_url,
        temperature=0.0,
        request_timeout=180.0,
    )


def _call_llm_with_retry(
    llm,
    prompt: str,
    retries: int = 3,
) -> str:
    """
    Call the LLM with retry support.
    """

    for attempt in range(retries):

        try:

            response = llm.invoke(prompt)

            return response.content.strip()

        except Exception as exc:

            logger.warning(
                "LLM call failed on attempt %d/%d: %s",
                attempt + 1,
                retries,
                exc,
            )

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    logger.error("All LLM retries failed.")

    return ""


# ============================================================================
# JSON PARSING
# ============================================================================

def clean_json_response(raw: str) -> str:
    """
    Remove markdown code fences and surrounding text where possible.
    """

    raw = raw.strip()

    raw = re.sub(
        r"```(?:json)?",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = raw.replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1:
        return raw[start:end + 1]

    return raw


def parse_json_response(raw: str) -> Dict[str, Any]:

    cleaned = clean_json_response(raw)

    try:

        return json.loads(cleaned)

    except Exception:

        logger.warning(
            "Could not parse JSON response: %r",
            raw,
        )

        return {}


def clamp_score(value: Any) -> float:

    try:

        value = float(value)

    except Exception:

        return 0.0

    return max(0.0, min(1.0, value))


# ============================================================================
# METRIC 1
# RETRIEVAL RELEVANCE
# ============================================================================

RETRIEVAL_JUDGE_PROMPT = """
You are evaluating one retrieved chunk from a Retrieval-Augmented Generation
(RAG) system.

Your task is to determine whether this chunk contains information that is
actually useful and sufficient to help answer the question correctly.

Do NOT judge based only on topic similarity.

A chunk is RELEVANT if:

1. It directly contains the answer, OR
2. It contains factual evidence necessary to derive the answer, OR
3. It contains an essential part of the answer.

A chunk is NOT RELEVANT if:

1. It only discusses the same broad topic,
2. It is merely semantically similar,
3. It contains unrelated information,
4. It does not help answer the specific question.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CHUNK:
{chunk}

Return ONLY valid JSON:

{{
    "relevant": 0 or 1,
    "score": 0.0 to 1.0,
    "reason": "short explanation"
}}
"""


def judge_retrieved_chunk(
    question: str,
    reference: str,
    chunk: str,
    llm,
) -> Dict[str, Any]:
    """
    Judge whether one retrieved chunk is relevant to answering the question.
    """

    prompt = RETRIEVAL_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        chunk=chunk,
    )

    raw = _call_llm_with_retry(llm, prompt)

    result = parse_json_response(raw)

    return {
        "relevant": int(result.get("relevant", 0)),
        "score": clamp_score(result.get("score", 0.0)),
        "reason": str(result.get("reason", "")),
    }


def metric_retrieval_relevance(
    question: str,
    reference: str,
    retrieved_docs: List[Dict],
    llm,
) -> Dict[str, Any]:
    """
    Evaluate every retrieved chunk individually.

    Returns:

        hit_at_k
        precision_at_k
        mean_relevance
        chunk-level judgments
    """

    if not retrieved_docs:

        return {
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "mean_relevance": 0.0,
            "chunk_judgments": [],
        }

    judgments = []

    for rank, doc in enumerate(retrieved_docs, start=1):

        judgment = judge_retrieved_chunk(
            question=question,
            reference=reference,
            chunk=doc["content"],
            llm=llm,
        )

        judgment["rank"] = rank
        judgment["distance"] = doc.get("distance")
        judgment["similarity"] = doc.get("similarity")

        judgments.append(judgment)

    relevant_flags = [
        j["relevant"]
        for j in judgments
    ]

    scores = [
        j["score"]
        for j in judgments
    ]

    hit_at_k = float(
        any(flag == 1 for flag in relevant_flags)
    )

    precision_at_k = float(
        np.mean(relevant_flags)
    ) if relevant_flags else 0.0

    mean_relevance = float(
        np.mean(scores)
    ) if scores else 0.0

    return {
        "hit_at_k": round(hit_at_k, 4),
        "precision_at_k": round(precision_at_k, 4),
        "mean_relevance": round(mean_relevance, 4),
        "chunk_judgments": judgments,
    }


# ============================================================================
# METRICS 2-4
# GROUNDNESS, CORRECTNESS, RELEVANCE
# ============================================================================

ANSWER_JUDGE_PROMPT = """
You are a strict evaluator for a Retrieval-Augmented Generation system.

Evaluate the GENERATED ANSWER on exactly three independent dimensions.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}


------------------------------------------------------------
1. GROUNDEDNESS
------------------------------------------------------------

Determine whether the factual claims made in the generated answer are
supported by the retrieved context.

Important:

- Every factual claim must be supported by the context.
- If the answer contains unsupported factual claims, reduce the score.
- If the answer contradicts the context, the score should be low.
- If the answer says it does not have enough information and makes no
  unsupported claims, groundedness can be high.
- Do not confuse correctness with groundedness.

------------------------------------------------------------
2. ANSWER CORRECTNESS
------------------------------------------------------------

Determine whether the generated answer conveys the same factual answer as
the reference answer.

Important:

- The wording does not need to be identical.
- Semantic equivalence is acceptable.
- The answer must contain the important facts required by the reference.
- Being on the same general topic is NOT enough.
- A vague answer should not receive full credit.
- An answer that says it lacks information when the reference contains the
  answer is incorrect.
- Missing important facts should reduce the score.
- Contradictory facts should reduce the score.

------------------------------------------------------------
3. ANSWER RELEVANCE
------------------------------------------------------------

Determine whether the generated answer directly answers the question.

Important:

- Being related to the topic is not enough.
- The answer must address what was specifically asked.
- A response that discusses surrounding concepts but does not answer the
  actual question should receive a low score.
- Concise answers can receive full credit if they directly answer the question.

------------------------------------------------------------

Return ONLY valid JSON:

{{
    "groundedness": 0.0,
    "answer_correctness": 0.0,
    "answer_relevance": 0.0,
    "reasoning": {{
        "groundedness": "short explanation",
        "answer_correctness": "short explanation",
        "answer_relevance": "short explanation"
    }}
}}
"""


def metric_llm_judges(
    question: str,
    reference: str,
    context: str,
    answer: str,
    llm,
) -> Dict[str, Any]:
    """
    Evaluate groundedness, answer correctness, and answer relevance.
    """

    prompt = ANSWER_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        context=context[:8000],
        answer=answer,
    )

    raw = _call_llm_with_retry(
        llm,
        prompt,
    )

    result = parse_json_response(raw)

    reasoning = result.get(
        "reasoning",
        {},
    )

    return {
        "groundedness": clamp_score(
            result.get("groundedness", 0.0)
        ),

        "answer_correctness": clamp_score(
            result.get("answer_correctness", 0.0)
        ),

        "answer_relevance": clamp_score(
            result.get("answer_relevance", 0.0)
        ),

        "reasoning": reasoning,
    }


# ============================================================================
# GENERATE ANSWER
# ============================================================================

def generate_answer(
    question: str,
    retriever: RAGRetriever,
    llm,
) -> tuple[str, str, List[Dict]]:
    """
    Retrieve documents, format context, and generate answer.

    Returns:

        answer
        formatted_context
        retrieved_documents
    """

    docs = retriever.retrieve(
        question,
        top_k=TOP_K,
    )

    context = retriever.format_context(
        docs,
    )

    if not docs:

        answer = (
            "I don't have enough relevant information "
            "to answer this question."
        )

    else:

        answer = ask(
            question,
            retriever,
            llm,
            top_k=TOP_K,
        )

    return answer, context, docs


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation():

    import pandas as pd

    logger.info("=" * 60)
    logger.info("RAG Evaluation Pipeline")
    logger.info("=" * 60)

    load_env()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH,
    )

    required_columns = {
        "Question",
        "Reference Answer",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    logger.info(
        "Loaded %d test cases from %s",
        len(df),
        DATASET_PATH,
    )

    # ------------------------------------------------------------
    # Initialize RAG components
    # ------------------------------------------------------------

    embedder = EmbeddingManager()

    store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )

    logger.info(
        "Total chunks in vector database: %d",
        store.collection.count(),
    )

    retriever = RAGRetriever(
        store,
        embedder,
    )

    llm = build_llm()

    # Log the effective URL being used
    active_url = getattr(llm, "base_url", OLLAMA_BASE_URL)
    logger.info(
        "LLM: %s @ %s",
        OLLAMA_MODEL,
        active_url,
    )

    # ------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------

    results: List[Dict[str, Any]] = []

    total = len(df)

    for idx, row in df.iterrows():

        question = str(
            row["Question"]
        ).strip()

        reference = str(
            row["Reference Answer"]
        ).strip()

        logger.info(
            "\n[%d/%d] Q: %s…",
            idx + 1,
            total,
            question[:100],
        )

        start_time = time.time()

        # --------------------------------------------------------
        # Generate answer
        # --------------------------------------------------------

        answer, context, docs = generate_answer(
            question,
            retriever,
            llm,
        )

        latency = round(
            time.time() - start_time,
            3,
        )

        logger.info(
            "  → Answer (%ss): %s…",
            latency,
            answer[:150],
        )

        for i, doc in enumerate(docs):
            logger.info(
                "  Context [%d] (sim: %.4f): %s...", 
                i+1, 
                doc.get("similarity") or 0.0, 
                doc["content"][:100].replace("\n", " ")
            )

        # --------------------------------------------------------
        # Retrieval evaluation
        # --------------------------------------------------------

        retrieval_metrics = metric_retrieval_relevance(
            question=question,
            reference=reference,
            retrieved_docs=docs,
            llm=llm,
        )

        logger.info(
            "  Hit@%d              : %.4f",
            TOP_K,
            retrieval_metrics["hit_at_k"],
        )

        logger.info(
            "  Retrieval Precision : %.4f",
            retrieval_metrics["precision_at_k"],
        )

        logger.info(
            "  Mean Chunk Relevance: %.4f",
            retrieval_metrics["mean_relevance"],
        )

        # --------------------------------------------------------
        # Answer evaluation
        # --------------------------------------------------------

        answer_metrics = metric_llm_judges(
            question=question,
            reference=reference,
            context=context,
            answer=answer,
            llm=llm,
        )

        logger.info(
            "  Groundedness        : %.4f",
            answer_metrics["groundedness"],
        )

        logger.info(
            "  Answer Correctness  : %.4f",
            answer_metrics["answer_correctness"],
        )

        logger.info(
            "  Answer Relevance    : %.4f",
            answer_metrics["answer_relevance"],
        )

        # --------------------------------------------------------
        # Save per-question result
        # --------------------------------------------------------

        results.append(
            {
                "index": int(idx),

                "question": question,

                "reference_answer": reference,

                "generated_answer": answer,

                "retrieved_chunks": [
                    {
                        "rank": i + 1,
                        "content": doc["content"],
                        "distance": doc.get("distance"),
                        "similarity": doc.get("similarity"),
                    }

                    for i, doc in enumerate(docs)
                ],

                "retrieval_evaluation": retrieval_metrics,

                "answer_evaluation": answer_metrics,

                "latency_seconds": latency,
            }
        )

        # Incremental save to ensure results aren't lost if the script crashes or is interrupted
        try:
            log_path.write_text(
                json.dumps({"partial_results": results}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to incrementally save results: %s", e)

        time.sleep(0.2)

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def aggregate_metric(
        values: List[float],
    ) -> Dict[str, float]:

        if not values:

            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
        }

    # ------------------------------------------------------------
    # Collect metric values
    # ------------------------------------------------------------

    hit_values = [
        r["retrieval_evaluation"]["hit_at_k"]
        for r in results
    ]

    precision_values = [
        r["retrieval_evaluation"]["precision_at_k"]
        for r in results
    ]

    relevance_values = [
        r["retrieval_evaluation"]["mean_relevance"]
        for r in results
    ]

    groundedness_values = [
        r["answer_evaluation"]["groundedness"]
        for r in results
    ]

    correctness_values = [
        r["answer_evaluation"]["answer_correctness"]
        for r in results
    ]

    answer_relevance_values = [
        r["answer_evaluation"]["answer_relevance"]
        for r in results
    ]

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {

        "eval_timestamp": timestamp,

        "embedding_model": EMBEDDING_MODEL,

        "llm": OLLAMA_MODEL,

        "num_test_cases": total,

        "top_k_retrieval": TOP_K,

        "aggregate": {

            "retrieval_hit_at_k": aggregate_metric(
                hit_values
            ),

            "retrieval_precision_at_k": aggregate_metric(
                precision_values
            ),

            "retrieval_mean_chunk_relevance": aggregate_metric(
                relevance_values
            ),

            "groundedness": aggregate_metric(
                groundedness_values
            ),

            "answer_correctness": aggregate_metric(
                correctness_values
            ),

            "answer_relevance": aggregate_metric(
                answer_relevance_values
            ),
        },

        "per_question_results": results,
    }

    # =========================================================================
    # SAVE JSON
    # =========================================================================

    log_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "\nFull results saved → %s",
        log_path,
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    csv_rows = []

    for r in results:

        csv_rows.append(
            {

                "index": r["index"],

                "question": r["question"],

                "reference_answer": r["reference_answer"],

                "generated_answer": r["generated_answer"],

                "retrieval_hit_at_k": (
                    r["retrieval_evaluation"]["hit_at_k"]
                ),

                "retrieval_precision_at_k": (
                    r["retrieval_evaluation"]["precision_at_k"]
                ),

                "retrieval_mean_chunk_relevance": (
                    r["retrieval_evaluation"]["mean_relevance"]
                ),

                "groundedness": (
                    r["answer_evaluation"]["groundedness"]
                ),

                "answer_correctness": (
                    r["answer_evaluation"]["answer_correctness"]
                ),

                "answer_relevance": (
                    r["answer_evaluation"]["answer_relevance"]
                ),

                "latency_seconds": r["latency_seconds"],
            }
        )

    if csv_rows:

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=csv_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(csv_rows)

    logger.info(
        "Summary CSV saved → %s",
        summary_path,
    )

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================

    aggregate = summary["aggregate"]

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)

    logger.info(
        "%-38s %7s  %7s  %7s  %7s",
        "Metric",
        "Mean",
        "Std",
        "Min",
        "Max",
    )

    logger.info("-" * 70)

    metric_names = [

        "retrieval_hit_at_k",

        "retrieval_precision_at_k",

        "retrieval_mean_chunk_relevance",

        "groundedness",

        "answer_correctness",

        "answer_relevance",
    ]

    for metric_name in metric_names:

        metric = aggregate[metric_name]

        logger.info(

            "%-38s %7.4f  %7.4f  %7.4f  %7.4f",

            metric_name,

            metric["mean"],

            metric["std"],

            metric["min"],

            metric["max"],
        )

    logger.info("=" * 70)

    return summary


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    run_evaluation()