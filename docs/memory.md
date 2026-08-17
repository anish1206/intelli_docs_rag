# Intelli Docs RAG - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Core RAG Components](#core-rag-components)
6. [FastAPI Backend](#fastapi-backend)
7. [Chat Memory & Session Management](#chat-memory--session-management)
8. [React Frontend](#react-frontend)
9. [RAG Pipeline](#rag-pipeline)
10. [Document Ingestion](#document-ingestion)
11. [Embedding Generation](#embedding-generation)
12. [Vector Store Management](#vector-store-management)
13. [Retrieval Pipeline](#retrieval-pipeline)
14. [LLM Integration](#llm-integration)
15. [Custom Evaluation Pipeline](#custom-evaluation-pipeline)
16. [Known Issues & Debugging](#known-issues--debugging)
17. [Production Considerations](#production-considerations)
18. [Setup & Usage](#setup--usage)
19. [Development Notes](#development-notes)

---

## Project Overview

**Project Name:** Intelli Docs RAG

**Purpose:** A comprehensive Retrieval-Augmented Generation (RAG) application for querying PDF, PPT, Word, and text documents using advanced document parsing, vector embeddings, and local/cloud LLM integration. Features a full-stack web UI with multi-session persistent chat.

**Key Features:**
- Multi-format document parsing using Docling (PDF, PPT, PPTX, DOC, DOCX, TXT)
- Semantic search with SentenceTransformer embeddings
- Persistent vector storage with ChromaDB (cosine similarity space)
- Enhanced retrieval with re-ranking and deduplication
- Memory-safe, page-by-page PDF loading to avoid `bad_alloc` errors
- Multiple LLM backends (Gemini, Ollama)
- **FastAPI REST backend** serving the RAG pipeline over HTTP
- **Multi-session chat** with Redis persistence (history + answer cache)
- **React + Vite frontend** with multi-chat sidebar UI
- Generic greeting interceptor (no RAG pipeline triggered for casual openers)
- Fully LLM-based evaluation pipeline with 4 metrics
- Source citation and context formatting

**Technology Stack:**
| Layer | Technology |
|---|---|
| Document Parsing | Docling (heavy ML features disabled for production) |
| Embeddings | SentenceTransformer (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Store | ChromaDB (persistent, cosine distance space) |
| LLM Backends | Ollama (`qwen2.5:7b` via Google Colab + Cloudflare tunnel) |
| Backend Framework | FastAPI + Uvicorn |
| Session Memory | Redis (conversation history + answer cache) |
| Frontend | React 18 + Vite |
| API Client | Fetch API (custom service layer) |
| Evaluation | Fully LLM-based judges (4 metrics) |
| Language | Python 3.12, JavaScript (ESM) |

---

## Current Status

### Completed Components

1. **RAG Pipeline (Fully Functional)**
   - ✅ PDF/Text document ingestion with Docling
   - ✅ Memory-safe page-by-page PDF loading (`rebuild_vector_store.py`)
   - ✅ Document chunking and metadata extraction
   - ✅ Embedding generation with SentenceTransformer
   - ✅ ChromaDB vector store with cosine similarity space
   - ✅ Enhanced retrieval pipeline with proper cosine re-ranking
   - ✅ Context formatting with source citations
   - ✅ `format_context()` alias for eval pipeline compatibility
   - ✅ Local LLM generation (Qwen/Ollama via Colab tunnel)
   - ✅ End-to-end RAG workflow

2. **FastAPI Backend (`backend/server.py`)**
   - ✅ `POST /chat` — main RAG endpoint with session-aware caching
   - ✅ `GET /health` — returns backend status and document count
   - ✅ `GET /sessions` — returns all chat sessions sorted by recency
   - ✅ `GET /chat/{session_id}/history` — returns message history for a session
   - ✅ `GET /` — root health message
   - ✅ CORS middleware for frontend communication
   - ✅ Session auto-registration on first message with generated title
   - ✅ Generic greeting interceptor (bypasses RAG pipeline)

3. **Multi-Session Chat Memory (`backend/chat_memory.py`)**
   - ✅ Session registration with timestamp + auto-generated title
   - ✅ `chat:sessions` Redis Sorted Set for ordered session listing
   - ✅ `chat:{session_id}:title` string key per session
   - ✅ `chat:{session_id}:history` Redis List per session
   - ✅ `chat:{session_id}:cache:{sha256}` answer cache per session
   - ✅ `get_all_sessions()` — sorted by most recent
   - ✅ `get_history(session_id)` — full conversation list
   - ✅ `add_message(role, content, session_id)` — push message
   - ✅ `get_cached_answer(question, session_id)` — SHA-256 cache lookup
   - ✅ `cache_answer(question, response, session_id)` — cache full response
   - ✅ `clear_history(session_id)` / `clear_cache(session_id)` — cleanup

4. **React Frontend (`frontend/`)**
   - ✅ Multi-session sidebar with "Recent Chats" list
   - ✅ Floating dark-theme sidebar toggle button (left edge, centered vertically)
   - ✅ Session switching — loads historical messages from backend
   - ✅ New Chat button (inside sidebar only)
   - ✅ Auto-load most recent session on app startup
   - ✅ Session list refresh after each message
   - ✅ Hero landing page + chat stream view
   - ✅ Sticky bottom input bar in chat mode
   - ✅ Loading animation with typing dots
   - ✅ Source citation cards (collapsible) in assistant messages
   - ✅ Backend health status badge in header
   - ✅ Error callout card with retry button

5. **Custom Evaluation Pipeline (Fully LLM-based — `eval/eval_rag_updated.py`)**
   - ✅ Retrieval Relevance (LLM judge, per-chunk, with Hit@K / Precision@K / Mean Relevance)
   - ✅ Groundedness metric (LLM judge)
   - ✅ Answer Correctness metric (LLM judge)
   - ✅ Answer Relevance metric (LLM judge)
   - ✅ Single combined LLM call for metrics 2–4 efficiency
   - ✅ Retry logic with exponential backoff
   - ✅ Robust JSON parsing (markdown fence stripping)
   - ✅ Result aggregation (mean/std/min/max) and logging
   - ✅ JSON and CSV output formats
   - ✅ Incremental JSON save after every question (crash-safe)
   - ✅ Retrieved chunks logged to console per question
   - ✅ `build_llm()` dynamically reads `OLLAMA_BASE_URL` from `.env` (supports remote Colab tunnel)
   - ✅ `request_timeout=180.0` added to prevent hangs

6. **Enhanced Retrieval System**
   - ✅ Cosine similarity scoring: `sim = 1.0 - float(dist)` (correct formula)
   - ✅ Hybrid search with over-fetching (top_k × 2)
   - ✅ Metadata filtering support
   - ✅ Deduplication by content hash
   - ✅ Enhanced scoring with heading overlap
   - ✅ Content length weighting
   - ✅ Score threshold filtering post-enhanced-scoring
   - ✅ Source citation formatting

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      React Frontend (Vite)                        │
│   Sidebar (Session List) → Chat Stream → Input Bar               │
└──────────────────────────────┬───────────────────────────────────┘
                               │  HTTP (Fetch API)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│               FastAPI Backend (backend/server.py)                  │
│   POST /chat  •  GET /sessions  •  GET /chat/{id}/history         │
└──────┬──────────────────────────┬────────────────────────────────┘
       │                          │
       ▼                          ▼
┌─────────────────┐    ┌─────────────────────────────────────────┐
│  Redis (Memory) │    │           RAG Pipeline (rag/)            │
│  Sessions       │    │  Retriever → ChromaDB → Ollama (LLM)    │
│  History        │    │  SentenceTransformer (Embeddings)        │
│  Answer Cache   │    └─────────────────────────────────────────┘
└─────────────────┘
```

### Request Flow: POST /chat

```
Frontend → POST /chat { question, session_id }
    ↓
1. Validate question (non-empty)
2. Session registration (auto-title from first 5 words)
3. Greeting interceptor check (bypass RAG for casual openers)
4. Redis cache lookup — if hit, return cached response immediately
    ↓ (cache miss)
5. RAGRetriever.retrieve(question)   → ChromaDB cosine search
6. retriever.format_context(docs)   → Citation-annotated text block
7. LLM.invoke(prompt)               → Ollama via Cloudflare tunnel
8. memory.add_message(user + assistant, session_id)
9. memory.cache_answer(question, result, session_id)
10. Return { question, answer, sources[], from_cache: false }
```

---

## Project Structure

```
inteli_docs_rag/
├── rag/                              # Core RAG package
│   ├── __init__.py                  # Public API exports
│   ├── config.py                    # Centralized configuration (reads .env)
│   └── pipeline.py                  # Core RAG logic: EmbeddingManager, VectorStore, RAGRetriever, ask()
│
├── backend/                         # FastAPI application
│   ├── server.py                    # FastAPI app, routes, LLM init, request handling
│   └── chat_memory.py               # Redis session management, conversation history, answer cache
│
├── frontend/                        # React + Vite web application
│   ├── src/
│   │   ├── App.jsx                  # Root component: session state, sidebar, chat layout
│   │   ├── App.css                  # Global styles, sidebar, floating toggle, chat UI
│   │   ├── index.css                # CSS variables & design tokens
│   │   ├── main.jsx                 # React entry point
│   │   ├── components/
│   │   │   ├── Header.jsx           # Top nav bar (logo + status badge only)
│   │   │   ├── ChatMessage.jsx      # Message bubble + source citation cards
│   │   │   ├── QuestionInput.jsx    # Text input + submit button
│   │   │   └── StatusBadge.jsx      # Backend health indicator pill
│   │   └── services/
│   │       └── api.js               # API service layer (checkBackendHealth, askQuestion, getSessions, getChatHistory)
│   ├── public/                      # Static assets (logo image)
│   ├── vite.config.js               # Vite config (proxy to localhost:8000)
│   └── package.json
│
├── eval/                            # Evaluation pipeline
│   ├── eval_rag_updated.py          # ✅ Current evaluation (LLM judges, 4 metrics)
│   ├── evaluate_rag.py              # Legacy evaluation script
│   ├── test_dataset.csv             # Evaluation questions (Question + Reference Answer)
│   └── eval_results_*.json          # Evaluation results
│
├── data/                            # Data directory
│   ├── pdf/                         # PDF documents
│   ├── text/                        # Text documents
│   └── vector_store/                # ChromaDB persistent storage
│
├── docs/                            # Documentation
│   ├── memory.md                    # This file
│   └── tp.txt                       # Scratch / notes
│
├── notebook/                        # Jupyter notebooks
│   ├── 1_document.ipynb
│   ├── 2_.ipynb
│   └── 3_docling.ipynb
│
├── chat.py                          # Interactive CLI interface
├── rebuild_vector_store.py          # Memory-safe vector store rebuild script
├── requirements.txt
├── .env                             # Secrets & config (gitignored)
├── .env.example                     # Template
└── package.json                     # Root-level (concurrently for dev)
```

---

## Core RAG Components

### 1. Configuration System (`rag/config.py`)

**Purpose:** Single source of truth for all tunable constants. Values are read from `.env` via `python-dotenv`.

**Key Configuration Parameters:**

```python
# Paths
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval Parameters
TOP_K = int(os.getenv("TOP_K", 5))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 4000))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.25))

# Document Ingestion
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# LLM Backends
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0.3))
```

> **Important:** `SCORE_THRESHOLD` is applied **after** `_enhanced_score()` is computed, not on the raw similarity.

### 2. Embedding Manager (`rag/pipeline.py`)

```python
class EmbeddingManager:
    def embed(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray
    def generate_embeddings(self, texts: List[str]) -> np.ndarray  # notebook alias
```

Model: `all-MiniLM-L6-v2` → 384-dimensional dense embeddings.

### 3. Vector Store (`rag/pipeline.py`)

ChromaDB with `"hnsw:space": "cosine"`. Returns distances in `[0, 2]`. Convert: `sim = 1.0 - dist`.

### 4. RAG Retriever (`rag/pipeline.py`)

```python
class RAGRetriever:
    def retrieve(self, query, top_k, score_threshold, filter_metadata) -> List[Dict]
    def format_context_for_llm(self, retrieved_docs, max_context_length) -> str
    def format_context(self, docs, max_chars) -> str  # eval alias
```

**Enhanced Ranking Logic:**
```python
sim = 1.0 - float(dist)              # correct for cosine space
h = hash(text)                        # dedup by content hash
enhanced_score = self._enhanced_score(text, query, sim, meta)
if enhanced_score < score_threshold:  # threshold after enhancement
    continue
```

**Enhanced Scoring:**
```python
def _enhanced_score(self, document, query, similarity_score, metadata) -> float:
    score = similarity_score
    # Heading/query word overlap boost
    overlap = len(q_terms & h_terms)
    if overlap > 0:
        score += 0.1 * overlap
    # Content length boost
    if len(document) > 1000: score += 0.1
    elif len(document) > 500: score += 0.05
    return min(score, 1.0)
```

### 5. High-Level `ask()` Function (`rag/pipeline.py`)

```python
def ask(query, retriever, llm, top_k, max_context_length, filter_metadata) -> str:
    docs = retriever.retrieve(query, top_k=top_k)
    if not docs:
        return "I don't have enough relevant information to answer this question."
    context = retriever.format_context_for_llm(docs, max_context_length)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
    return llm.invoke(prompt).content
```

---

## FastAPI Backend

**File:** `backend/server.py`  
**Run command:** `uvicorn backend.server:app --reload` (from project root)

### Startup Initialization

On startup, the server:
1. Loads `EmbeddingManager` (SentenceTransformer model)
2. Loads `VectorStore` (connects to ChromaDB)
3. Creates `RAGRetriever`
4. Initializes `ChatOllama` LLM pointing to `OLLAMA_BASE_URL` from `.env`
5. Initializes `ChatMemory` (connects to Redis on `localhost:6379`)

### Request Model

```python
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"  # Unique session UUID from frontend
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root — "API is running" |
| `GET` | `/health` | Status + `vector_store_documents` count |
| `GET` | `/sessions` | All sessions, sorted most-recent first |
| `GET` | `/chat/{session_id}/history` | Message list for a session |
| `POST` | `/chat` | Main RAG endpoint |

### POST /chat — Processing Steps

```
1. Strip & validate question
2. session_id = request.session_id
3. Register session (auto-title from first 5 words of question)
4. Greeting intercept: if question in {"hi", "hello", "hey", ...}
       → store to history, return canned response, skip RAG
5. Cache check: memory.get_cached_answer(question, session_id)
       → if hit, return cached response with from_cache: True
6. retriever.retrieve(question)
7. retriever.format_context(docs)
8. LLM prompt assembly & llm.invoke(prompt)
9. memory.add_message(user, session_id)
10. memory.add_message(assistant, session_id)
11. memory.cache_answer(question, result, session_id)
12. Return { question, answer, sources[], from_cache: false }
```

### Generic Greeting Interceptor

The server intercepts casual openers **before** touching the RAG pipeline:

```python
greetings = {"hi", "hello", "hey", "greetings", "good morning", "good evening", "howdy", "sup"}
if question.lower() in greetings:
    answer = "Hello! I am your Intelli Docs assistant. How can I help you with your documents today?"
    memory.add_message(role="user", content=question, session_id=session_id)
    memory.add_message(role="assistant", content=answer, session_id=session_id)
    return {"question": question, "answer": answer, "sources": [], "from_cache": False}
```

---

## Chat Memory & Session Management

**File:** `backend/chat_memory.py`  
**Store:** Redis on `localhost:6379`, DB 0, `decode_responses=True`

### Redis Key Schema

| Key | Type | Contents |
|---|---|---|
| `chat:sessions` | Sorted Set | `{session_id: timestamp}` — all session IDs sorted by time |
| `chat:{session_id}:title` | String | Short auto-generated title (first 5 words of first message) |
| `chat:{session_id}:history` | List | JSON-encoded `{role, content}` messages in order |
| `chat:{session_id}:cache:{sha256}` | String | JSON-encoded full RAG response for a question |

### API

```python
class ChatMemory:
    # Session management
    def register_session(self, session_id: str, title: str)
    def get_all_sessions(self) -> List[{id, title, timestamp}]

    # Conversation history
    def add_message(self, role: str, content: str, session_id: str = "default")
    def get_history(self, session_id: str = "default") -> List[{role, content}]
    def clear_history(self, session_id: str = "default")

    # Answer cache
    def cache_answer(self, question: str, response: dict, session_id: str = "default")
    def get_cached_answer(self, question: str, session_id: str = "default") -> dict | None
    def clear_cache(self, session_id: str = "default")
```

### Cache Key Generation

```python
def _question_key(self, question: str, session_id: str) -> str:
    normalized = " ".join(question.lower().split())
    question_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"chat:{session_id}:cache:{question_hash}"
```

Cache keys are **per-session** — the same question in two different sessions produces two separate cache entries.

---

## React Frontend

**Framework:** React 18 + Vite  
**Run command:** `npm run dev` (from `frontend/`)

### Component Tree

```
App.jsx                       ← Root: session state, sidebar state, submit logic
├── floating-sidebar-toggle   ← Fixed button on left edge (visible when sidebar closed)
├── <aside class="sidebar">   ← Collapsible panel (slide in from left)
│   ├── sidebar-header        ← "Recent Chats" + close (✕) button
│   ├── new-chat-btn          ← Message-Square-Plus icon + "New Chat"
│   └── sessions-list         ← Session items (active highlight, click to switch)
└── <div class="main-wrapper">
    ├── Header.jsx             ← Logo + StatusBadge only (no toggle, no new chat)
    ├── <main class="main-content">
    │   ├── hero-landing-wrapper  ← Shown when no messages (isLanding = true)
    │   └── chat-stream           ← Shown when messages exist
    │       ├── ChatMessage.jsx   ← Per message (user bubble / assistant bubble + sources)
    │       ├── loading-bubble    ← Typing animation (3 dots + status text)
    │       └── error-callout-card
    └── <footer class="app-footer-input-bar">  ← Sticky bottom bar (chat mode only)
        └── QuestionInput.jsx
```

### State Management (`App.jsx`)

```javascript
const [healthStatus, setHealthStatus] = useState('checking');  // 'checking' | 'healthy' | 'offline'
const [docCount, setDocCount] = useState(0);
const [question, setQuestion] = useState('');
const [messages, setMessages] = useState([]);        // Current session's messages
const [isLoading, setIsLoading] = useState(false);
const [errorMsg, setErrorMsg] = useState(null);
const [sessionId, setSessionId] = useState(Date.now().toString());  // UUID-ish
const [sessions, setSessions] = useState([]);        // All sessions from /sessions
const [isSidebarOpen, setIsSidebarOpen] = useState(false);
```

### Session Lifecycle

| Action | What happens |
|---|---|
| App mount | `initializeApp()` → health check + `GET /sessions` → load most recent session history |
| User sends message | `POST /chat` with `session_id` → backend auto-registers session → `GET /sessions` refresh |
| Click "New Chat" | `setMessages([])`, `setSessionId(Date.now())` — fresh session, empty context |
| Click session in sidebar | `GET /chat/{id}/history` → `setMessages(mappedHistory)` |
| Sidebar toggle | `isSidebarOpen` flips → CSS slide transition |

### API Service Layer (`frontend/src/services/api.js`)

```javascript
export async function checkBackendHealth()  // GET /health
export async function askQuestion(question, sessionId = "default")  // POST /chat
export async function getSessions()  // GET /sessions
export async function getChatHistory(sessionId)  // GET /chat/{sessionId}/history
```

### UI Layout: Sidebar Behavior

The sidebar slides in from the left using `margin-left: -260px` when hidden and `margin-left: 0` when open. The main wrapper fills the remaining width via `flex: 1`.

A **floating dark-theme toggle button** is fixed to `left: 0`, vertically centered (`top: 50%`), with a `›` chevron icon. It is only rendered when `!isSidebarOpen`. Hovering slightly extends it to the right as a visual affordance.

```css
.floating-sidebar-toggle {
  position: fixed;
  top: 50%;
  left: 0;
  transform: translateY(-50%);
  background-color: var(--color-deep-midnight);
  border: 1px solid var(--color-faded-steel);
  border-left: none;
  border-radius: 0 8px 8px 0;
  padding: 16px 8px;
  /* ... hover: padding-left increases to 12px */
}
```

---

## RAG Pipeline

### Complete Workflow

1. **Document Ingestion** — `rebuild_vector_store.py`
   - Lightweight Docling converter (OCR off, all heavy ML features off)
   - PDFs: page-by-page via `PdfReader` + `page_range` kwarg
   - Non-PDF: default `DoclingLoader`

2. **Embedding Generation** — `EmbeddingManager`
   - `all-MiniLM-L6-v2` → 384-dim dense float32 vectors

3. **Vector Storage** — `VectorStore` (ChromaDB)
   - `hnsw:space: cosine` collection metadata
   - Metadata flattening before insert

4. **Query Processing** — `RAGRetriever.retrieve()`
   - Query embedded → cosine search (over-fetch top_k × 2)
   - `sim = 1.0 - dist`
   - Deduplicate → enhanced score → threshold → rank → top_k

5. **Context Formatting** — `format_context_for_llm()`
   - Source citation header `[Source: file.pdf, Page N, Section: ...]`
   - Length management (`MAX_CONTEXT_CHARS`)

6. **LLM Generation** — `llm.invoke(prompt)` via Ollama
   - Prompt includes context + question + instructions
   - Returns answer with source citations

---

## Document Ingestion

### Memory-Safe PDF Loading (`rebuild_vector_store.py`)

**Problem:** `std::bad_alloc` from RapidOCR on large PDFs.

**Solution:** Lightweight `DocumentConverter` with all heavy ML features disabled, reused across all files. PDFs processed one page at a time.

```python
def build_light_pdf_converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions(
        force_backend_text=True,
        do_ocr=False,                    # Main memory saver
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=False,
        document_timeout=120,
        ocr_batch_size=1,
        layout_batch_size=1,
    )
    ...
```

```python
def load_pdf_documents(file_path, converter):
    page_count = len(PdfReader(str(file_path)).pages)
    for page_no in range(1, page_count + 1):
        loader = DoclingLoader(..., convert_kwargs={"page_range": (page_no, page_no)})
        docs = loader.load()
        ...
        gc.collect()
```

---

## Embedding Generation

**Model:** `all-MiniLM-L6-v2` — 384-dimensional dense embeddings.

---

## Vector Store Management

**Collection Name:** `pdf_documents` (configurable via `.env`)  
**Persist Directory:** `data/vector_store`  
**Distance Space:** `cosine` — distances in `[0, 2]`. Use `sim = 1.0 - dist`.

### Metadata Flattening

ChromaDB only accepts `str | int | float | bool | None` metadata values. Docling generates nested dicts/lists, which are flattened before insertion:
- `origin.filename` → `filename`
- `headings` list → comma-joined string
- `dl_meta.doc_items[0].prov[0].page_no` → `page_no`
- Other complex types → `json.dumps(value)`

---

## Retrieval Pipeline

### Similarity Score History

| Version | Formula | Status |
|---|---|---|
| Initial | `sim = 1 - distance` on l2 space | ❌ All filtered out |
| Fix 1 | `sim = float(-dist)` | ⚠️ Works for ranking, wrong semantics |
| **Current** | `sim = 1.0 - float(dist)` + `hnsw:space=cosine` | ✅ Correct |

---

## LLM Integration

### Ollama (Current Production Backend)

**Model:** `qwen2.5:7b` running on **Google Colab GPU** exposed via **Cloudflare tunnel**.

```python
llm = ChatOllama(
    model=OLLAMA_MODEL,        # qwen2.5:7b
    base_url=OLLAMA_BASE_URL,  # e.g. https://xxxx.trycloudflare.com
    temperature=0.0,
)
```

**Tunnel setup (Colab notebook):**
1. Install & start Ollama, pull model
2. Run `cloudflared tunnel --url http://localhost:11434` (or ngrok)
3. Copy the public URL into `.env` as `OLLAMA_BASE_URL`
4. Restart the FastAPI backend

> **If you see `httpx.ConnectError: getaddrinfo failed`** — the tunnel URL in `.env` has expired. Restart the Colab cell, copy the new URL, update `.env`, restart uvicorn.

### Gemini (Evaluation & Optional)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=GEMINI_TEMPERATURE,
    api_key=os.getenv("GEMINI_API_KEY")
)
```

---

## Custom Evaluation Pipeline

> **Active file:** `eval/eval_rag_updated.py`

### Design Philosophy

Fully LLM-based evaluation using `qwen2.5:7b` via Ollama on Colab. Dataset requires only `Question` and `Reference Answer` columns.

### Metrics

1. **Retrieval Relevance** — per-chunk LLM judge → Hit@K, Precision@K, Mean Chunk Relevance
2. **Groundedness** — are answer claims supported by context?
3. **Answer Correctness** — does answer match reference semantically?
4. **Answer Relevance** — does answer address the question asked?

Metrics 2–4 are judged in a single combined LLM call for efficiency.

### Output Files

- `eval/eval_results_<timestamp>.json` — full per-question results
- `eval/eval_summary_<timestamp>.csv` — per-question scores
- Incremental saves after every question (crash-safe)

---

## Known Issues & Debugging

### Issue 1: Ollama Tunnel Expired
**Error:** `httpx.ConnectError: [Errno 11001] getaddrinfo failed`  
**Cause:** The Cloudflare/Ngrok tunnel URL in `.env` has expired or the Colab session ended.  
**Fix:** Restart the Colab tunnel cell, copy the new URL, update `OLLAMA_BASE_URL` in `.env`, restart uvicorn.

### Issue 2: Metadata Type Error (Resolved)
**Error:** `ValueError: Expected metadata value to be a str, int, float, bool, or None`  
**Fix:** Metadata flattening in `VectorStore.add_documents()`.

### Issue 3: Memory Allocation Errors (Resolved)
**Error:** `std::bad_alloc` from Docling/RapidOCR  
**Fix:** `build_light_pdf_converter()` + page-by-page loading in `rebuild_vector_store.py`.

### Issue 4: Similarity Score Bug (Resolved)
**Fix:** `sim = 1.0 - float(dist)` with `hnsw:space=cosine` collection.

### Issue 5: Enhanced Score Not Applied (Resolved)
**Fix:** `_enhanced_score()` now called explicitly; threshold applied after enhanced scoring.

### Issue 6: Module Not Found Running Server
**Error:** `ModuleNotFoundError: No module named 'backend'` or `No module named 'rag'`  
**Fix:** Always run from the **project root**: `uvicorn backend.server:app --reload`. Do NOT run from inside the `backend/` folder.

### Debugging Checklist for Empty Retrieval

1. Check collection has documents: `store.collection.count()`
2. Check raw distances: print `distances` from results
3. Verify `hnsw:space` is `cosine`: `store.collection.metadata`
4. Temporarily set `SCORE_THRESHOLD=0.0` to confirm retrieval works
5. Print `sim = 1.0 - dist` — should be `~0.3–0.9` for good matches

---

## Production Considerations

### Scalability

- **Document Processing:** Single-threaded; parallelize with Celery/worker pool for large sets
- **Vector Store:** ChromaDB suitable for small-medium datasets; Qdrant/Weaviate/Pinecone for large scale
- **Session Storage:** Redis is in-memory; configure `appendonly yes` for disk persistence across restarts

### Performance

- Batch embedding with progress bars for large document sets
- GPU acceleration for SentenceTransformer
- Cache frequent query embeddings
- Tune HNSW parameters for faster ANN search

---

## Setup & Usage

### Installation

```bash
git clone <repository-url>
cd inteli_docs_rag
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set OLLAMA_BASE_URL and GEMINI_API_KEY
```

### Start Redis

```bash
# Windows: download Redis for Windows or use WSL
redis-server
```

### Start Backend

```bash
# Always from project root
uvicorn backend.server:app --reload
```

### Start Frontend

```bash
cd frontend
npm install    # first time only
npm run dev
```

### Rebuild Vector Store

```bash
python rebuild_vector_store.py             # Wipe & rebuild
python rebuild_vector_store.py --keep-existing  # Append mode
python rebuild_vector_store.py --data-dir /path/to/docs
```

### Run Evaluation

```bash
python eval/eval_rag_updated.py
```

Dataset at `eval/test_dataset.csv` must have columns: `Question`, `Reference Answer`.

---

## Development Notes

### Environment Variables (`.env`)

```
# LLM
OLLAMA_BASE_URL=https://<tunnel-url>
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TEMPERATURE=0.0
GEMINI_API_KEY=<your-key>
GEMINI_MODEL=gemini-2.5-pro
GEMINI_TEMPERATURE=0.3

# Vector Store
COLLECTION_NAME=pdf_documents
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Retrieval
TOP_K=5
MAX_CONTEXT_CHARS=4000
SCORE_THRESHOLD=0.25
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'rag'` | Not running from project root | `cd inteli_docs_rag` then run uvicorn |
| `httpx.ConnectError: getaddrinfo failed` | Tunnel URL expired | Restart Colab tunnel, update `OLLAMA_BASE_URL` in `.env`, restart uvicorn |
| `ConnectionRefusedError` (Redis) | Redis not running | `redis-server` |
| `sqlite3.OperationalError: database is locked` | Multiple processes accessing ChromaDB | Kill zombie processes |
| All retrieved docs empty | Wrong distance formula or threshold too high | Check `hnsw:space=cosine`, set `SCORE_THRESHOLD=0.0` to debug |
| Eval LLM times out | Model on Colab too slow / tunnel dropped | Increase `request_timeout`, restart Colab + tunnel |
| Frontend shows "Backend Offline" | FastAPI not running or wrong port | `uvicorn backend.server:app --reload` |
| Sessions not appearing in sidebar | Redis not running | `redis-server` |

### Code Organization Principles

- **Single Source of Truth:** All config in `rag/config.py` (reads `.env`)
- **Session Isolation:** Every Redis key is namespaced by `session_id`
- **Cache Before LLM:** Every `/chat` request checks Redis before invoking the LLM
- **Greeting Interceptor:** Casual messages bypass the RAG pipeline entirely
- **Eval Compatibility:** `format_context()` alias keeps eval pipeline decoupled from method renames
- **Module Imports:** Always run server from project root so `rag` and `backend` packages resolve

### Future Enhancements

1. **Streaming Responses:** Real-time SSE streaming from LLM to frontend
2. **Hybrid Search:** Combine semantic + BM25 keyword search
3. **Session Deletion:** Allow users to delete sessions from the sidebar
4. **Document Upload UI:** Drag-and-drop ingestion directly from the web UI
5. **User Authentication:** Multi-user support with session isolation
6. **Persistent Redis:** Configure AOF for history survival across Redis restarts
7. **Answer Feedback Loop:** Thumbs up/down to improve retrieval quality

---

**Document Version:** 3.0  
**Last Updated:** August 17, 2026  
**Status:** Fully updated — FastAPI backend, multi-session Redis, React sidebar UI, Cloudflare tunnel integration, greeting interceptor
