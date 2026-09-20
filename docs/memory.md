Here is the updated, comprehensive, and exhaustive `memory.md`. It documents the entire system from core RAG fundamentals to the newly implemented LangGraph cyclical state machine, native tool calling, CRAG retrieval self-correction, in-loop hallucination verification, and the automated `pytest` test suite.

***

# `memory.md` — Intelli Docs Agentic RAG: Architecture & Technical Source of Truth

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Current System Status](#2-current-system-status)
3. [System Architecture & Topology](#3-system-architecture--topology)
4. [Project Directory Structure](#4-project-directory-structure)
5. [Configuration & Environment Variables](#5-configuration--environment-variables)
6. [Core RAG Subsystem (`rag/`)](#6-core-rag-subsystem-rag)
7. [Agentic Engine & LangGraph Workflow (`rag/agent/`)](#7-agentic-engine--langgraph-workflow-ragagent)
8. [Tool Registry & Execution Subsystem](#8-tool-registry--execution-subsystem)
9. [FastAPI Backend & Orchestration (`backend/server.py`)](#9-fastapi-backend--orchestration-backendserverpy)
10. [Chat Memory & Redis Session Management (`backend/chat_memory.py`)](#10-chat-memory--redis-session-management-backendchat_memorypy)
11. [React Frontend & Contract Guarantee (`frontend/`)](#11-react-frontend--contract-guarantee-frontend)
12. [Document Ingestion & Vector Storage](#12-document-ingestion--vector-storage)
13. [Testing, Verification & CI Infrastructure](#13-testing-verification--ci-infrastructure)
14. [Offline Evaluation Pipeline (`eval/`)](#14-offline-evaluation-pipeline-eval)
15. [Debugging, Common Pitfalls & Operational Playbook](#15-debugging-common-pitfalls--operational-playbook)
16. [Production Considerations & Future Roadmap](#16-production-considerations--future-roadmap)

---

## 1. Project Overview

**Project Name:** Intelli Docs Agentic RAG  
**Purpose:** A high-performance, cost-free, production-grade Agentic Document Assistant optimized for academic and enterprise multi-format documents (PDF, PPT, Word, TXT). The system transitions beyond static vector lookup into an autonomous reasoning loop featuring conversational query rewriting, Corrective RAG (CRAG) with dynamic query reformulation, deterministic tool execution (sandboxed mathematics and document catalog queries), and an in-loop hallucination guard that validates generated responses against retrieved facts before rendering.

### Key Architectural Pillars
* **Hybrid Zero-Cost Topology:** Local orchestrator (FastAPI, Redis, ChromaDB, SentenceTransformer) paired with remote GPU inference (Qwen 2.5:7b on Google Colab via Cloudflare/ngrok tunnels).
* **Cyclical Agentic State Machine:** Powered by LangGraph, enabling adaptive branching, tool calling, retrieval grading, self-correction, and reflection.
* **Conversational Context Resolution:** Resolves ambiguous pronouns and follow-up turns into fully qualified standalone queries using sliding-window Redis history.
* **CRAG (Corrective RAG):** Evaluates retrieved chunk relevance and automatically reformulates queries when document signal is poor.
* **Deterministic Tool Layer:** Sandboxed local math evaluation (`simpleeval`) for formulas/grades and direct metadata inspection tools that bypass vector search.
* **In-Loop Groundedness Reflection:** Validates model drafts against factual citations prior to client egress, preventing hallucinations.
* **Deterministic Fast Paths:** Instant routing for greetings and SHA-256 Redis answer caching to avoid remote LLM latency.
* **Full-Stack Persistence:** Multi-session conversation management via Redis and an interactive, citation-aware React 18 frontend.
* **Comprehensive Test Suite:** Fully offline, deterministic `pytest` framework spanning unit, integration, and graph boundary conditions.

### Technology Stack
| Layer | Component | Version / Model | License / Cost | Role |
|---|---|---|---|---|
| **Agent Framework** | LangGraph | `>=0.2.0` | MIT / Free | Cyclical state graph, conditional routing, loop circuit breaker |
| **Backend API** | FastAPI + Uvicorn | `0.110+` | MIT / Free | REST orchestration, async dispatch, request validation |
| **LLM Inference** | Ollama (`qwen2.5:7b`) | Remote (Colab GPU) | Apache 2.0 / Free | Tool calling, routing, synthesis, grading, reflection |
| **Tunnelling** | Cloudflare Tunnel / ngrok | Latest CLI | Free Tier | Secure HTTPS exposure of Colab `localhost:11434` |
| **Embeddings** | SentenceTransformer | `all-MiniLM-L6-v2` (384-d) | Apache 2.0 / Free | Dense vector generation for documents and queries |
| **Vector Database** | ChromaDB | Persistent Local HNSW | Apache 2.0 / Free | Cosine distance index (`hnsw:space: cosine`) |
| **Memory / Cache** | Redis | Server `>=6.0` | BSD / Free | Session tracking, conversation history, exact query cache |
| **Document Parsing**| Docling | Lightweight Engine | MIT / Free | Memory-safe, page-by-page streaming extraction |
| **Safe Math Tool** | Simpleeval | `>=0.9.13` | MIT / Free | Sandboxed mathematical expression evaluation |
| **Frontend UI** | React 18 + Vite | ESM / Node 18+ | MIT / Free | Multi-session sidebar, citation cards, status indicators |
| **Testing Suite** | Pytest + Fakeredis | Latest | MIT / Free | Offline mock-driven unit and integration test framework |

---

## 2. Current System Status

### ✅ Completed & Fully Verified Milestones

1. **Core Vector Engine & Ingestion (`rag/`)**
   - Memory-safe PDF document streaming via `DoclingLoader` with all heavy ML/OCR models disabled (`std::bad_alloc` immune).
   - Document chunking (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`) with structural metadata extraction.
   - ChromaDB persistence with cosine vector space (`sim = 1.0 - float(dist)`).
   - Metadata normalization and flattening (JSON strings for complex nested items).
   - Enhanced retrieval scoring incorporating heading word overlap and chunk length bonuses.

2. **Agentic Subsystem (`rag/agent/`)**
   - **`AgentState` TypedDict:** Strongly typed state tracking queries, tools, retry counts, document chunks, and verification flags.
   - **Multi-Turn Context Rewriter:** Evaluates sliding window Redis history to rewrite ambiguous questions into standalone queries.
   - **Agent Router:** Dispatches to vector search, math evaluation, metadata listing, or direct answer based on intent.
   - **Tool Ecosystem:**
     - `search_notes`: Over-fetching semantic retrieval via ChromaDB.
     - `calculate`: Sandboxed, exception-safe arithmetic evaluation via `simpleeval`.
     - `list_available_notes`: Direct catalog inspection of indexed documents without vector search.
   - **CRAG Grader & Reformulator:** Inspects chunk relevance and triggers search query reformulation if chunks lack answering signal (capped at 1 retry).
   - **Synthesizer:** Merges citations (`[Source: file, Page N]`) and tool calculations into structured markdown responses.
   - **In-Loop Hallucination Guard:** Pre-egress reflection checking if drafted claims are strictly grounded in retrieved context (capped at 1 retry).
   - **LangGraph StateGraph:** Cyclical assembly connecting nodes, conditional decision gates, and circuit breakers.

3. **Backend & Session Orchestration (`backend/`)**
   - Pre-graph fast path 1: Instant canned greeting interceptor (zero LLM latency).
   - Pre-graph fast path 2: Exact per-session SHA-256 query cache lookup (instant retrieval from Redis).
   - Synchronous invocation of compiled LangGraph workflow inside `POST /chat`.
   - Redis-backed sliding-window history extraction (`get_recent_history_formatted`).
   - Session tracking, dynamic title generation, and reverse-chronological session listing.

4. **Frontend Compatibility (`frontend/`)**
   - Additive response envelope: UI consumes `{ question, answer, sources, from_cache, agent_metadata }` seamlessly.
   - Zero UI regressions; existing layout, collapsible citation pills, sidebar session switching, and health badges function unaltered.
   - Production build verified (`npm run build` exits with code 0).

5. **Testing Infrastructure (`tests/` & `eval/`)**
   - Root `pytest.ini` with automated `pythonpath` discovery.
   - 100% offline unit tests covering configuration, Redis memory, agent tools, nodes, state graphs, and FastAPI routes via `fakeredis` and `unittest.mock`.
   - Standalone CLI validation suite (`eval/test_agent.py`) for live environment testing across all agentic paths.
   - Offline multi-metric LLM-as-a-judge evaluation suite (`eval/eval_rag_updated.py`).

---

## 3. System Architecture & Topology

### 3.1 Global Topology: Local Core vs. Remote Inference

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               LOCAL MACHINE (Host System)                               │
│                                                                                        │
│  ┌────────────────────────┐         HTTP (Fetch)         ┌──────────────────────────┐  │
│  │  React 18 UI (Vite)    │ ◄──────────────────────────► │  FastAPI Server (8000)   │  │
│  │  Sidebar • Chat Stream │                              │  backend/server.py       │  │
│  └────────────────────────┘                              └─────────────┬────────────┘  │
│                                                                        │               │
│                                  ┌─────────────────────────────────────┴────────────┐  │
│                                  │                                                  │  │
│                                  ▼                                                  ▼  │
│                     ┌──────────────────────────┐                       ┌─────────────────────┐
│                     │       Redis (6379)       │                       │   LangGraph Agent   │
│                     │  Sessions • History      │                       │     rag/agent/      │
│                     │  SHA-256 Answer Cache    │                       └──────────┬──────────┘
│                     └──────────────────────────┘                                  │    │
│                                                                                   │    │
│            ┌──────────────────────────────────────────────────────────────────────┘    │
│            │                                                                           │
│            ▼                                      ▼                                    │
│  ┌─────────────────────────┐            ┌───────────────────┐                          │
│  │   ChromaDB (Persistent) │            │   Simpleeval      │                          │
│  │   Cosine Vector Store   │            │   Safe Math Engine│                          │
│  └─────────────▲───────────┘            └───────────────────┘                          │
│                │                                                                       │
│  ┌─────────────┴───────────┐                                                           │
│  │  SentenceTransformer    │                                                           │
│  │  all-MiniLM-L6-v2 (384) │                                                           │
│  └─────────────────────────┘                                                           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTPS Requests (ChatOllama LangChain Client)
                                            │ Public Encrypted Tunnel
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            REMOTE INFERENCE (Google Colab GPU)                         │
│                                                                                        │
│  ┌───────────────────────────┐                           ┌──────────────────────────┐  │
│  │ Cloudflare / ngrok Tunnel │ ◄───────────────────────► │   Ollama Server (11434)  │  │
│  │ (e.g. *.trycloudflare.com)│                           │   Model: qwen2.5:7b      │  │
│  └───────────────────────────┘                           └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 LangGraph Execution Lifecycle (`POST /chat`)

```
                          [ Client Request: { question, session_id } ]
                                                │
                                                ▼
                              [ Validate Input: question.strip() ]
                                                │
                                                ▼
                              [ Redis Session Registration & Title ]
                                                │
                                                ▼
                                  /───────────────────────────\
                                 <   Is Greeting Statement?    > ── Yes ──► [ Instant Return: Canned ]
                                  \───────────────────────────/
                                                │ No
                                                ▼
                                  /───────────────────────────\
                                 <    Exact Cache Match in     > ── Yes ──► [ Instant Return: Cached ]
                                 <   Redis (SHA-256 Hash)?     >            (from_cache: True)
                                  \───────────────────────────/
                                                │ No (Cache Miss)
                                                ▼
                              [ Extract Sliding History (Redis) ]
                              [ Build Initial AgentState Payload ]
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Node: rewrite_query     │
                                  │  (Resolve pronouns/terms) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │       Node: router        │
                                  │ (search, calc, docs, chat)│
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                     /────────────────────────────────────────────────────\
                    <                   tool_choice?                       >
                     \────────────────────────────────────────────────────/
                       │                       │                        │
         "calculate"   │         "list_docs"   │            "search"    │        "direct"
                       ▼                       ▼                        ▼           │
             ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐  │
             │ Node: calculate │     │ Node: list_docs  │     │Node: search_notes│  │
             │ (simpleeval)    │     │ (doc catalogue)  │     │(ChromaDB Top-K)  │  │
             └────────┬────────┘     └─────────┬────────┘     └─────────┬────────┘  │
                      │                        │                        │           │
                      │                        │                        ▼           │
                      │                        │              ┌──────────────────┐  │
                      │                        │              │Node: grade_docs  │  │
                      │                        │              │ (CRAG Relevance) │  │
                      │                        │              └─────────┬────────┘  │
                      │                        │                        │           │
                      │                        │               /─────────────────\  │
                      │                        │              <   Docs Relevant?  > │
                      │                        │               \─────────────────/  │
                      │                        │                 │ No         │ Yes │
                      │                        │                 │(retry < 1) │     │
                      │                        │                 ▼            │     │
                      │                        │       ┌───────────────────┐  │     │
                      │                        │       │ Node: reformulate │  │     │
                      │                        │       │  (Keyword tweak)  │  │     │
                      │                        │       └─────────┬─────────┘  │     │
                      │                        │                 │            │     │
                      │                        │                 └──────►─────┤     │
                      │                        │                              │     │
                      ▼                        ▼                              ▼     ▼
             ┌──────────────────────────────────────────────────────────────────────┐
             │                     Node: generate_answer                            │
             │          (Synthesize answer + citations from context)                │
             └──────────────────────────────────┬───────────────────────────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Node: hallucination_guard │
                                  │  (Check claim grounding)  │
                                  └─────────────┬─────────────┘
                                                │
                                 /─────────────────────────────\
                                <    Hallucination Detected?    >
                                 \─────────────────────────────/
                                   │ Yes (retry < 1)         │ No (or cap reached)
                                   ▼                         ▼
                         [ Trigger Regeneration ]       [ Mark Grounded = True ]
                         [ With Strict Grounding]            │
                                   │                         ▼
                                   └────────────►     [ Graph State Output ]
                                                             │
                                                             ▼
                                              [ Save Turn to Redis History ]
                                              [ Write Response to Redis Cache ]
                                                             │
                                                             ▼
                                              [ Return JSON to Frontend ]
```

---

## 4. Project Directory Structure

```
inteli_docs_rag/
├── rag/                                # Core Engine & Agent Package
│   ├── __init__.py                     # Package exports: EmbeddingManager, VectorStore, RAGRetriever, ask
│   ├── config.py                       # Centralized configuration & environment loader
│   ├── pipeline.py                     # Document embedding, ChromaDB vector store, RAGRetriever
│   └── agent/                          # Agentic Subsystem (LangGraph)
│       ├── __init__.py                 # Exports: build_agentic_rag_graph, AgentState, tools
│       ├── state.py                    # TypedDict schema for AgentState
│       ├── tools.py                    # Tools: calculate (simpleeval), search_notes, list_available_notes
│       ├── prompts.py                  # Strict prompt templates & markdown-fence JSON parsers
│       ├── nodes.py                    # Execution node factories and conditional routing predicates
│       └── graph.py                    # LangGraph StateGraph assembly and workflow compilation
│
├── backend/                            # FastAPI Application
│   ├── server.py                       # FastAPI entrypoint, lifecycle init, route dispatching
│   └── chat_memory.py                  # Redis session indexing, sliding-window history, answer cache
│
├── frontend/                           # React 18 + Vite Web Application
│   ├── src/
│   │   ├── App.jsx                     # Root application container (layout, session state, submit logic)
│   │   ├── App.css                     # Global dark-theme styling, sidebar transitions, floating toggle
│   │   ├── index.css                   # CSS tokens, color schemes, typography
│   │   ├── main.jsx                    # React DOM mount entry point
│   │   ├── components/
│   │   │   ├── Header.jsx              # Application header with live backend status badge
│   │   │   ├── ChatMessage.jsx         # Message bubble with collapsible citation cards
│   │   │   ├── QuestionInput.jsx       # Autosizing input field and submit button
│   │   │   └── StatusBadge.jsx         # Backend connectivity pill (Healthy / Offline)
│   │   └── services/
│   │       └── api.js                  # Frontend Fetch client (health, chat, sessions, history)
│   ├── public/                         # Static web assets (branding logos, icons)
│   ├── vite.config.js                  # Vite configuration with API proxy to localhost:8000
│   └── package.json                    # Node dependencies and build scripts
│
├── tests/                              # Automated Pytest Suite (100% Offline via Mocks)
│   ├── __init__.py                     # Test package initializer
│   ├── conftest.py                     # Pytest fixtures: fakeredis client, mock retriever, mock LLM
│   ├── test_config.py                  # Configuration parameter parsing & env override unit tests
│   ├── test_chat_memory.py             # ChatMemory & get_recent_history_formatted unit tests
│   ├── test_agent_tools.py             # Tools unit tests: calculate arithmetic & note catalog
│   ├── test_agent_nodes.py             # Agent node execution & conditional decision gate unit tests
│   ├── test_agent_graph.py             # End-to-end graph traversal & conditional path tests
│   ├── test_server.py                  # FastAPI TestClient endpoint integration tests
│   └── test_e2e_agent.py               # Deterministic end-to-end agentic behavior verification
│
├── eval/                               # Evaluation & Live Verification Suite
│   ├── test_agent.py                   # Standalone CLI runner for testing against running server
│   ├── eval_rag_updated.py             # Multi-metric offline LLM-as-a-judge evaluation harness
│   ├── evaluate_rag.py                 # Legacy evaluation harness
│   ├── test_dataset.csv                # Ground truth test set (Question, Reference Answer)
│   └── eval_results_*.json             # Historical evaluation artifacts
│
├── data/                               # Data Directory
│   ├── pdf/                            # Source PDF notes and documents
│   ├── text/                           # Source raw text and markdown files
│   └── vector_store/                   # Persistent ChromaDB parquet/index store
│
├── docs/                               # Project Documentation
│   └── memory.md                       # This document (Single Source of Truth)
│
├── notebook/                           # Prototyping & Docling Experimentation Notebooks
│   ├── 1_document.ipynb
│   ├── 2_.ipynb
│   └── 3_docling.ipynb
│
├── pytest.ini                          # Pytest configuration (pythonpath, paths, options)
├── rebuild_vector_store.py             # Memory-safe PDF document ingestion script
├── requirements.txt                    # Unified Python dependencies (Free / Open Source)
├── .env                                # Local secrets & runtime configuration (Git ignored)
├── .env.example                        # Template for required environment variables
└── package.json                        # Root package manifest (Concurrently scripts)
```

---

## 5. Configuration & Environment Variables

### 5.1 Configuration Architecture (`rag/config.py`)
All parameters are managed through `rag/config.py`, loading variables dynamically via `python-dotenv`. Default values ensure the application boots out of the box in development environments.

```python
# System Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Vector Store
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")

# Retrieval Tunables
TOP_K = int(os.getenv("TOP_K", 5))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 4000))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.25))

# Document Ingestion
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Remote Ollama LLM Settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))

# Agentic Execution Controls
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", 1))
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", 0.5))
HISTORY_WINDOW_TURNS = int(os.getenv("HISTORY_WINDOW_TURNS", 3))

# Optional Gemini Fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0.3))
```

### 5.2 Environment Variable Matrix (`.env.example`)
```env
# ==========================================
# LLM Inference (Google Colab via Cloudflare)
# ==========================================
OLLAMA_BASE_URL=https://your-tunnel-subdomain.trycloudflare.com
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TEMPERATURE=0.0

# ==========================================
# Agentic RAG Control Parameters
# ==========================================
AGENT_MAX_RETRIES=1
CRAG_RELEVANCE_THRESHOLD=0.5
HISTORY_WINDOW_TURNS=3

# ==========================================
# Vector Database & Embeddings
# ==========================================
COLLECTION_NAME=pdf_documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=5
MAX_CONTEXT_CHARS=4000
SCORE_THRESHOLD=0.25
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# ==========================================
# Optional External Providers
# ==========================================
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-pro
GEMINI_TEMPERATURE=0.3
```

---

## 6. Core RAG Subsystem (`rag/`)

### 6.1 Embedding Manager (`EmbeddingManager`)
Uses `sentence-transformers/all-MiniLM-L6-v2` running on CPU or local GPU:
* Produces dense embeddings of dimension $D = 384$.
* Normalizes output vectors to unit length ($\ell_2$), facilitating cosine distance calculations.
* Provides `embed(texts)` and an alias `generate_embeddings(texts)` for notebook compatibility.

### 6.2 Vector Store (`VectorStore`)
Maintains a local, persistent ChromaDB instance at `data/vector_store`:
* Collection initialized with metadata `{"hnsw:space": "cosine"}`.
* **Cosine Distance Space:** Raw distance output $d \in [0, 2]$.
* **Similarity Conversion:** $S_{\text{sim}} = 1.0 - \text{float}(d)$.
* **Metadata Normalization:** ChromaDB requires flat scalar metadata (`str`, `int`, `float`, `bool`). Any nested dictionaries or lists generated during document parsing are serialized to JSON strings prior to ingestion.

### 6.3 Enhanced RAG Retriever (`RAGRetriever`)
Executes an advanced multi-stage filtering and scoring pipeline:
1. **Over-fetching:** Queries ChromaDB for $K_{\text{fetch}} = \text{TOP\_K} \times 2$ chunks.
2. **Deduplication:** Computes an MD5/SHA content hash for each chunk body, discarding duplicates.
3. **Enhanced Scoring Formula:**
   $$\text{Score}_{\text{enhanced}} = \min\left(1.0, S_{\text{sim}} + B_{\text{heading}} + B_{\text{length}}\right)$$
   * $B_{\text{heading}} = 0.1 \times |\text{QueryTerms} \cap \text{HeadingTerms}|$ (rewards chunks whose section titles match the search query).
   * $B_{\text{length}} = +0.10 \text{ if } \text{length} > 1000 \text{ chars else } (+0.05 \text{ if } \text{length} > 500)$.
4. **Post-Score Threshold Filtering:** Chunks with $\text{Score}_{\text{enhanced}} < \text{SCORE\_THRESHOLD}$ are pruned.
5. **Ranking & Slicing:** Sorts remaining chunks in descending score order and slices to `top_k`.
6. **Context Citation Formatting:** Formats retrieved chunks into an LLM-ready context block:
   ```
   [Source: filename.pdf, Page 12, Section: Introduction]
   <chunk content text>
   ```

---

## 7. Agentic Engine & LangGraph Workflow (`rag/agent/`)

### 7.1 State Machine Schema (`rag/agent/state.py`)
All nodes in the LangGraph workflow communicate by mutating the centralized `AgentState`:

```python
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    question: str                      # Raw incoming question from user
    session_id: str                    # Current chat session identifier
    history: str                       # Formatted sliding-window history string
    standalone_query: str              # Context-resolved search query
    tool_choice: Optional[str]         # 'search', 'calculate', 'list_docs', 'direct'
    tool_output: Optional[Any]         # Execution result from invoked tool
    documents: List[Dict[str, Any]]    # Retrieved chunks from ChromaDB
    retrieval_retry_count: int         # Counter for CRAG query reformulations
    hallucination_retry_count: int     # Counter for guard regeneration loops
    is_relevant: bool                  # CRAG chunk relevance grade
    is_grounded: bool                  # In-loop hallucination guard validation flag
    final_answer: str                  # Generated response body
    sources: List[Dict[str, Any]]      # Citations list passed to frontend
```

### 7.2 Structured Prompt Engineering & Parsers (`rag/agent/prompts.py`)
Because Qwen 2.5:7b runs over an external tunnel, prompts are strictly token-optimized and use compact JSON payloads. All outputs pass through `clean_json_response()` to strip markdown fences (` ```json `):

```python
import re
import json

def clean_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback regex extraction for corrupted JSON strings
        if '"relevant": true' in cleaned.lower():
            return {"relevant": True}
        if '"grounded": true' in cleaned.lower():
            return {"grounded": True}
        return {}
```

#### Core Prompts:
1. **`REWRITE_PROMPT`:** Contextualizes follow-ups:
   > *"Given the chat history and the latest user question, rewrite the question into a clear, standalone search query. Do NOT answer it. Return ONLY the rewritten query."*
2. **`ROUTER_PROMPT`:** Intent classification:
   > *"Classify the user intent into one of: 'search' (searching notes/syllabus), 'calculate' (math/formulas/numbers), 'list_docs' (asking what files/notes exist), or 'direct' (general questions). Output strictly JSON: {\"tool\": \"<choice>\"}"*
3. **`CRAG_GRADER_PROMPT`:** Chunk signal verification:
   > *"Evaluate if the retrieved documents contain sufficient factual signal to answer the question. Output strictly JSON: {\"relevant\": true|false}"*
4. **`REFORMULATE_PROMPT`:** Query transformation for failed retrieval:
   > *"The initial document search failed. Provide 2-4 alternative keywords or search phrases targeting academic lecture notes. Output strictly the refined query."*
5. **`GUARD_PROMPT`:** Pre-egress reflection:
   > *"Check if every factual claim in the drafted answer is directly supported by the context. Output strictly JSON: {\"grounded\": true|false}"*

### 7.3 Workflow Nodes & Decisions (`rag/agent/nodes.py`)
Each node is created via closure/factory pattern passing runtime dependencies:

* **`rewrite_query_node`:** If `history` is empty, sets `standalone_query = question`. Otherwise invokes `REWRITE_PROMPT` to resolve pronouns (e.g. *"What is its runtime?"* $\to$ *"What is the runtime of QuickSort?"*).
* **`router_node`:** Invokes `ROUTER_PROMPT`, updating `tool_choice`.
* **`search_notes_node`:** Invokes `search_notes(retriever, standalone_query)`, setting `documents` and extracting initial `sources`.
* **`calculate_node`:** Extracts expression and executes sandboxed math via `calculate(expression)`. Sets `final_answer` and `tool_output`.
* **`list_docs_node`:** Calls `list_available_notes(retriever)` to return indexed files into `final_answer`.
* **`grade_retrieval_node`:** Runs `CRAG_GRADER_PROMPT`. Sets `is_relevant: bool`.
* **`reformulate_query_node`:** Increments `retrieval_retry_count += 1` and updates `standalone_query`.
* **`generate_answer_node`:** Renders context block from `documents`, binds with prompt template, and invokes LLM.
* **`hallucination_guard_node`:** Validates drafted answer against context. If ungrounded and `hallucination_retry_count < AGENT_MAX_RETRIES`, sets `is_grounded = False` and increments counter; otherwise sets `is_grounded = True`.

#### Conditional Edge Predicates:
* `route_decision(state) -> str`: Returns `tool_choice` string.
* `crag_decision(state) -> str`: Returns `"proceed"` if `is_relevant is True` or `retrieval_retry_count >= AGENT_MAX_RETRIES`; else `"retry"`.
* `guard_decision(state) -> str`: Returns `"pass"` if `is_grounded is True` or `hallucination_retry_count >= AGENT_MAX_RETRIES`; else `"regenerate"`.

### 7.4 Graph Assembly & Compilation (`rag/agent/graph.py`)
Compiles the complete workflow into an executable `CompiledStateGraph`:

```python
from langgraph.graph import StateGraph, END
from rag.agent.state import AgentState
from rag.agent import nodes

def build_agentic_rag_graph(retriever, llm):
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("rewrite_query", nodes.create_rewriter_node(llm))
    workflow.add_node("router", nodes.create_router_node(llm))
    workflow.add_node("search_notes", nodes.create_search_node(retriever))
    workflow.add_node("calculate", nodes.calculate_node)
    workflow.add_node("list_docs", nodes.create_list_docs_node(retriever))
    workflow.add_node("grade_retrieval", nodes.create_grade_node(llm))
    workflow.add_node("reformulate_query", nodes.create_reformulate_node(llm))
    workflow.add_node("generate_answer", nodes.create_generator_node(llm, retriever))
    workflow.add_node("hallucination_guard", nodes.create_guard_node(llm))

    # Connect Edges
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "router")

    workflow.add_conditional_edges(
        "router",
        nodes.route_decision,
        {
            "search": "search_notes",
            "calculate": "calculate",
            "list_docs": "list_docs",
            "direct": "generate_answer",
        }
    )

    workflow.add_edge("search_notes", "grade_retrieval")
    workflow.add_conditional_edges(
        "grade_retrieval",
        nodes.crag_decision,
        {
            "proceed": "generate_answer",
            "retry": "reformulate_query",
        }
    )
    workflow.add_edge("reformulate_query", "search_notes")

    workflow.add_edge("calculate", "generate_answer")
    workflow.add_edge("list_docs", "generate_answer")
    workflow.add_edge("generate_answer", "hallucination_guard")

    workflow.add_conditional_edges(
        "hallucination_guard",
        nodes.guard_decision,
        {
            "pass": END,
            "regenerate": "generate_answer",
        }
    )

    return workflow.compile()
```

---

## 8. Tool Registry & Execution Subsystem

Located in `rag/agent/tools.py`.

### 8.1 `calculate(expression: str) -> str`
* **Security & Sandboxing:** Implemented using `simpleeval.simple_eval`. It forbids `__import__`, `eval`, `exec`, OS calls, and attribute traversal.
* **Operators Supported:** Addition, subtraction, multiplication, division, modulo, exponents, bitwise shifts, and built-in math functions (`abs`, `round`, `min`, `max`).
* **Error Handling:** Traps `ZeroDivisionError`, `SyntaxError`, and `InvalidExpression`, returning a human-readable string: `"Error: Division by zero"` or `"Error: Invalid mathematical expression"`.

### 8.2 `search_notes(retriever, query: str, top_k: int = 5) -> List[Dict[str, Any]]`
* Directly wraps `retriever.retrieve(query=query, top_k=top_k)`.
* Returns structured chunk payloads with text, source filename, page number, section heading, and computed cosine relevance scores.

### 8.3 `list_available_notes(retriever) -> List[str]`
* Inspects `retriever.vector_store.collection` metadata directly using `.get(include=["metadatas"])`.
* Aggregates and extracts unique, sorted filenames currently indexed in the vector store.
* Bypasses vector embedding and similarity search entirely, guaranteeing 100% precision for catalog queries (*"What documents are indexed?"*).

---

## 9. FastAPI Backend & Orchestration (`backend/server.py`)

### 9.1 Server Startup Lifecycle
On invocation (`uvicorn backend.server:app --reload`), the application executes startup initialization:
1. Instantiates `EmbeddingManager` (loads `all-MiniLM-L6-v2`).
2. Connects to `VectorStore` (ChromaDB at `data/vector_store`).
3. Instantiates `RAGRetriever`.
4. Connects to `ChatMemory` (local Redis on `localhost:6379`).
5. Configures remote `ChatOllama` LLM pointing to `OLLAMA_BASE_URL` with `request_timeout=180.0`.
6. Compiles the LangGraph application: `agent_app = build_agentic_rag_graph(retriever, llm)`.

### 9.2 API Endpoint Specifications

| Method | Path | Request Body | Response Schema | Description |
|---|---|---|---|---|
| `GET` | `/` | None | `{"message": str}` | Root health check |
| `GET` | `/health` | None | `{"status": str, "vector_store_documents": int}` | Backend health & document count |
| `GET` | `/sessions` | None | `[{"id": str, "title": str, "timestamp": float}]` | Sessions sorted most recent first |
| `GET` | `/chat/{session_id}/history`| None | `[{"role": str, "content": str}]` | Full message log for a session |
| `POST`| `/chat` | `{"question": str, "session_id": str}` | See payload structure below | Main RAG & Agentic entry point |

### 9.3 Detailed Processing Logic for `POST /chat`
1. **Input Normalization:** Strips query whitespace. Rejects empty queries with `HTTPException(400)`.
2. **Session Initialization:** If session is new, auto-registers title in Redis using first 5 words of question.
3. **Deterministic Fast Path 1 (Greetings):** Checks against `{"hi", "hello", "hey", "greetings", "good morning", "howdy"}`. If matched:
   - Appends user and assistant messages to Redis history.
   - Returns instant canned response without LLM or vector search.
4. **Deterministic Fast Path 2 (Exact Answer Cache):** Checks `chat:{session_id}:cache:{sha256(question)}` in Redis. If hit:
   - Returns cached payload instantly with `from_cache: True`.
5. **Agent State Graph Dispatch (Cache Miss):**
   - Extracts sliding-window history: `memory.get_recent_history_formatted(session_id, max_turns=3)`.
   - Initializes `AgentState` dictionary.
   - Executes `agent_app.invoke(initial_state)`.
6. **State Unpacking & Persistence:**
   - Appends user query and generated `final_answer` to Redis conversation history.
   - Caches response in Redis under SHA-256 key.
7. **Response Serialization:** Emits JSON response matching frontend expectations:
   ```json
   {
     "question": "What is the worst case complexity of QuickSort?",
     "answer": "According to the lecture notes, QuickSort has a worst-case time complexity of O(n^2)...",
     "sources": [
       {
         "filename": "Algorithms_Lecture_4.pdf",
         "page_no": 14,
         "score": 0.88,
         "headings": "Sorting Algorithms, QuickSort Analysis"
       }
     ],
     "from_cache": false,
     "agent_metadata": {
       "standalone_query": "What is the worst case time complexity of QuickSort?",
       "tool_choice": "search",
       "retrieval_retried": false,
       "guard_passed": true
     }
   }
   ```

---

## 10. Chat Memory & Redis Session Management (`backend/chat_memory.py`)

Operates against Redis DB 0 on `localhost:6379` (`decode_responses=True`).

### 10.1 Key Schema Table
| Key Format | Redis Type | Value Schema / Purpose |
|---|---|---|
| `chat:sessions` | Sorted Set | `session_id` mapped to epoch timestamp score for time-sorted listing |
| `chat:{session_id}:title` | String | UTF-8 plain text title of the session (e.g. "QuickSort Worst Case Analysis") |
| `chat:{session_id}:history` | List | Chronological JSON strings: `{"role": "user"\|"assistant", "content": "..."}` |
| `chat:{session_id}:cache:{sha256}` | String | JSON-serialized complete response payload for exact-match caching |

### 10.2 Sliding-Window Conversational Extraction
```python
def get_recent_history_formatted(self, session_id: str, max_turns: int = None) -> str:
    """Extracts recent conversation formatted as User/Assistant dialogue for prompt context."""
    if max_turns is None:
        from rag.config import HISTORY_WINDOW_TURNS
        max_turns = HISTORY_WINDOW_TURNS

    history = self.get_history(session_id)
    if not history:
        return ""

    # Each turn represents a user + assistant pair
    recent_messages = history[-(max_turns * 2):]
    formatted = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages]
    return "\n".join(formatted)
```

### 10.3 Per-Session SHA-256 Hashing
Cache keys are strictly session-isolated:
$$\text{HashKey} = \text{SHA256}(\text{lowercase}(\text{normalize\_whitespace}(\text{question})))$$
$$\text{RedisKey} = \text{"chat:"} + \text{session\_id} + \text{":cache:"} + \text{HashKey}$$

---

## 11. React Frontend & Contract Guarantee (`frontend/`)

### 11.1 Component Tree & UI Architecture
```
App.jsx (Global Session & Chat State Manager)
├── floating-sidebar-toggle (Visible when sidebar collapsed; fixed at left center)
├── <aside className="sidebar"> (Collapsible slide-in panel)
│   ├── sidebar-header ("Recent Chats" title + Close button)
│   ├── new-chat-btn ("+ New Chat" session initiator)
│   └── sessions-list (Scrollable session list with active highlight)
└── <div className="main-wrapper">
    ├── Header.jsx (Branding logo + Live backend StatusBadge pill)
    ├── <main className="main-content">
    │   ├── hero-landing-wrapper (Displayed when conversation history is empty)
    │   └── chat-stream (Virtual chat scroll container)
    │       ├── ChatMessage.jsx (Message bubbles + collapsible citation source pills)
    │       ├── loading-bubble (3-dot bouncing typing indicator with status text)
    │       └── error-callout-card (Error message with inline retry trigger)
    └── <footer className="app-footer-input-bar"> (Sticky bottom container)
        └── QuestionInput.jsx (Autosizing textarea + submit button)
```

### 11.2 Backward Compatibility Guarantee
The React client API service layer (`frontend/src/services/api.js`) expects:
```javascript
const response = await fetch('/chat', { ... });
const data = await response.json();
// data: { question, answer, sources, from_cache }
```
The addition of `agent_metadata` from the agentic upgrade is **strictly non-breaking**. The frontend consumes `answer` and `sources` exactly as before. The application build passes cleanly:
```bash
cd frontend && npm run build
# vite build -> dist/ assets generated with zero errors
```

---

## 12. Document Ingestion & Vector Storage

### 12.1 Memory-Safe PDF Ingestion (`rebuild_vector_store.py`)
To prevent `std::bad_alloc` crashes during heavy OCR/vision processing on large PDF notes, ingestion uses a lightweight Docling pipeline:

```python
def build_light_pdf_converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions(
        force_backend_text=True,
        do_ocr=False,                      # Completely disables heavy OCR models
        do_table_structure=True,          # Retains Markdown table representations
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        generate_page_images=False,
        generate_parsed_pages=False,
        document_timeout=120,
    )
    ...
```

### 12.2 Page-by-Page Streaming Extraction
Rather than loading an entire 500-page book into memory, the builder streams pages iteratively:
1. `PdfReader(file_path).pages` determines total page count.
2. Iterates $P \in [1, \text{TotalPages}]$, instantiating `DoclingLoader` with `page_range=(P, P)`.
3. Splits extracted page text into chunks via `RecursiveCharacterTextSplitter`.
4. Executes explicit garbage collection (`gc.collect()`) after every page to flush intermediate buffers.
5. Embeds chunks via `EmbeddingManager` and flushes to ChromaDB.

---

## 13. Testing, Verification & CI Infrastructure

### 13.1 Pytest Infrastructure Setup (`pytest.ini`)
Configured to discover tests across the root repository:
```ini
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
```

### 13.2 Automated Test Suite (`tests/`)
All tests in `tests/` run **100% offline** without requiring live Redis, ChromaDB, or Google Colab instances:

| Test File | Target Subsystem | Scenarios Covered |
|---|---|---|
| `test_config.py` | `rag/config.py` | Default values verification; type conversions (`int`, `float`) for env overrides |
| `test_chat_memory.py` | `backend/chat_memory.py` | `fakeredis` isolation; history formatting; window slicing; empty sessions |
| `test_agent_tools.py` | `rag/agent/tools.py` | `calculate()` arithmetic, order of operations, division by zero; `list_available_notes()` metadata catalog |
| `test_agent_nodes.py` | `rag/agent/nodes.py` | Mocked LLM node outputs; `clean_json_response()` markdown fence stripping; retry counter increments |
| `test_agent_graph.py` | `rag/agent/graph.py` | Graph structure compilation; conditional edge routing (`search`, `calculate`, `list_docs`, `retry`) |
| `test_server.py` | `backend/server.py` | FastAPI `TestClient`; greeting fast path; cache hit path; full `/chat` agent dispatch |
| `test_e2e_agent.py` | Complete Agent Loop | Multi-turn pronoun rewrite; CRAG retry cap adherence; Hallucination guard loop cap adherence |

Run all tests:
```bash
pytest
```

### 13.3 Live Verification CLI Script (`eval/test_agent.py`)
Used to validate the live environment against a running FastAPI backend (`http://localhost:8000`):
```bash
python eval/test_agent.py --base-url http://localhost:8000
```
**Scenarios Executed:**
1. **Greeting Fast-Path Test:** Verifies instant return without LLM call.
2. **Multi-Turn Context Resolution:** Sends Turn 1 (*"What is QuickSort?"*), then Turn 2 (*"What is its worst case?"*). Verifies `agent_metadata.standalone_query` correctly contextualized the subject.
3. **Calculator Tool Test:** Queries weighted grade arithmetic. Verifies `agent_metadata.tool_choice == 'calculate'` and exact numeric output.
4. **Metadata Catalog Test:** Queries available documents. Verifies `agent_metadata.tool_choice == 'list_docs'`.
5. **CRAG & Hallucination Guard Test:** Validates syllabus question response with source citations and verified `guard_passed == True`.
6. **Answer Cache Hit Test:** Re-sends previous question; verifies instant response with `from_cache == True`.

---

## 14. Offline Evaluation Pipeline (`eval/`)

Located in `eval/eval_rag_updated.py`.

### 14.1 Philosophy & Execution
Evaluates RAG and Agent quality against `eval/test_dataset.csv` using Qwen 2.5:7b as an automated judge:
* Reads `OLLAMA_BASE_URL` dynamically from `.env`.
* Configures `request_timeout=180.0` with exponential backoff retries.
* Saves evaluation results incrementally to disk after every row to prevent data loss if the tunnel drops.

### 14.2 Evaluated Metrics
1. **Retrieval Relevance (Hit@K, Precision@K, Mean Score):** Evaluates retrieved document chunks individually for answering power.
2. **Groundedness (Faithfulness):** LLM judge determines whether claims in the generated response are directly supported by the context.
3. **Answer Correctness:** Compares semantic alignment against reference ground-truth text.
4. **Answer Relevance:** Checks whether the response directly addresses the user's inquiry.

*Metrics 2, 3, and 4 are bundled into a single JSON evaluation call to minimize remote inference round-trips.*

---

## 15. Debugging, Common Pitfalls & Operational Playbook

### Issue 1: Remote Colab Tunnel Disconnected / Expired
* **Symptom:** `httpx.ConnectError: [Errno 11001] getaddrinfo failed` or HTTP 502/504 errors in FastAPI console.
* **Root Cause:** Cloudflare Tunnel or Google Colab session timed out or restarted, changing the public URL.
* **Resolution:**
  1. Go to your active Google Colab tab.
  2. Re-run the tunnel cell: `cloudflared tunnel --url http://localhost:11434`.
  3. Copy the newly assigned HTTPS URL (`https://<subdomain>.trycloudflare.com`).
  4. Update `OLLAMA_BASE_URL` in `.env`.
  5. Restart Uvicorn: `uvicorn backend.server:app --reload`.

### Issue 2: Redis Server Unreachable
* **Symptom:** `redis.exceptions.ConnectionError: Error 10061 connecting to localhost:6379`.
* **Root Cause:** Redis service is not running locally.
* **Resolution:** Start the local Redis service:
  ```bash
  # Windows (WSL or Native Windows Redis port)
  redis-server
  # Linux / macOS
  brew services start redis  # or sudo systemctl start redis
  ```

### Issue 3: Ingestion Memory Crash (`std::bad_alloc`)
* **Symptom:** Ingestion process crashes abruptly when processing large PDF files.
* **Root Cause:** RapidOCR or layout vision models running out of system RAM.
* **Resolution:** Ensure `rebuild_vector_store.py` uses `build_light_pdf_converter()` (`do_ocr=False`) and page-by-page streaming with `gc.collect()`.

### Issue 4: ChromaDB Schema Type Errors
* **Symptom:** `ValueError: Expected metadata value to be a str, int, float, bool, or None`.
* **Root Cause:** Docling generates nested dictionary trees or list structures for headings and provenance.
* **Resolution:** Ensure metadata flattening in `VectorStore.add_documents()` serializes complex values into JSON strings before insertion.

### Issue 5: Markdown Code-Fence JSON Parsing Failure
* **Symptom:** Router, Grader, or Guard nodes falling back to defaults due to `JSONDecodeError`.
* **Root Cause:** Qwen 2.5:7b wrapping JSON responses in ` ```json ... ``` ` fences.
* **Resolution:** Always wrap raw LLM string responses in `clean_json_response()` before invoking `json.loads()`.

---

## 16. Production Considerations & Future Roadmap

### 16.1 Scalability & Production Hardening
* **Redis Persistence:** Configure Redis Append-Only File (`appendonly yes`) in `redis.conf` so chat history survives host reboots.
* **Vector Store Scaling:** ChromaDB is ideal for single-node deployments (<1M chunks). For enterprise scale, swap `VectorStore` with Qdrant, Milvus, or pgvector.
* **Streaming Tokens:** Upgrade `POST /chat` from standard JSON response to Server-Sent Events (SSE) using LangGraph's `.astream_events()` for real-time word-by-word streaming in the React UI.

### 16.2 Future Enhancements Roadmap
1. **Hybrid Keyword + Vector Search:** Combine BM25 sparse keyword matching with dense SentenceTransformer vectors via Reciprocal Rank Fusion (RRF).
2. **Document Upload via Web UI:** Drag-and-drop file ingestion directly in the React frontend with background Celery processing.
3. **User Authentication & RBAC:** Multi-tenant session separation via JWT authentication.
4. **Session Deletion & Export:** Provide UI controls to delete chat sessions or export conversations to Markdown/PDF notes.

---

## 17. Setup, Installation & Developer Guide

### 17.1 Initial Environment Setup
```bash
# 1. Clone repository and navigate to root
git clone <repo-url>
cd inteli_docs_rag

# 2. Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
# or: venv\Scripts\activate    # Windows

# 3. Install unified requirements
pip install -r requirements.txt

# 4. Configure local environment variables
cp .env.example .env
# Edit .env: Set OLLAMA_BASE_URL to your active Cloudflare tunnel URL
```

### 17.2 Document Indexing (Rebuild Vector Store)
```bash
# Place your PDF documents into data/pdf/
python rebuild_vector_store.py
```

### 17.3 Running Test Suite
```bash
# Execute offline automated unit and integration tests
pytest
```

### 17.4 Running the Full-Stack Application
```bash
# Terminal 1: Start Redis Server
redis-server

# Terminal 2: Start FastAPI Backend (Always run from project root!)
uvicorn backend.server:app --reload

# Terminal 3: Start React Frontend
cd frontend
npm install   # First time only
npm run dev
```

The React web application will be accessible at: `http://localhost:5173`  
The FastAPI Swagger documentation will be accessible at: `http://localhost:8000/docs`

***

**Document Version:** 4.0 (Agentic RAG Upgrade)  
**Maintained By:** Intelli Docs Engineering Team  
**Status:** Complete, Verified, and Production-Ready