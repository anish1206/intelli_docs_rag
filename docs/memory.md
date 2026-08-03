# Intelli Docs RAG - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [RAG Pipeline](#rag-pipeline)
7. [Document Ingestion](#document-ingestion)
8. [Embedding Generation](#embedding-generation)
9. [Vector Store Management](#vector-store-management)
10. [Retrieval Pipeline](#retrieval-pipeline)
11. [LLM Integration](#llm-integration)
12. [Custom Evaluation Pipeline](#custom-evaluation-pipeline)
13. [Known Issues & Debugging](#known-issues--debugging)
14. [Production Considerations](#production-considerations)
15. [Setup & Usage](#setup--usage)
16. [Development Notes](#development-notes)

---

## Project Overview

**Project Name:** Intelli Docs RAG

**Purpose:** A comprehensive Retrieval-Augmented Generation (RAG) application for querying PDF, PPT, Word, and text documents using advanced document parsing, vector embeddings, and local/cloud LLM integration.

**Key Features:**
- Multi-format document parsing using Docling (PDF, PPT, PPTX, DOC, DOCX, TXT)
- Semantic search with SentenceTransformer embeddings
- Persistent vector storage with ChromaDB (cosine similarity space)
- Enhanced retrieval with re-ranking and deduplication
- Memory-safe, page-by-page PDF loading to avoid `bad_alloc` errors
- Multiple LLM backends (Gemini, Ollama)
- Fully LLM-based evaluation pipeline with 4 metrics
- Interactive CLI chat interface
- Source citation and context formatting

**Technology Stack:**
- **Document Parsing:** Docling (heavy ML features disabled for production use)
- **Embeddings:** SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB (persistent, cosine distance space)
- **LLM Backends:** Google Gemini (gemini-2.5-flash), Ollama (qwen2.5:7b via Google Colab + Ngrok tunnel)
- **Framework:** LangChain
- **Evaluation:** Fully LLM-based judges (4 metrics)
- **Language:** Python 3.x

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
   - ✅ Local LLM generation (Qwen/Ollama)
   - ✅ Cloud LLM integration (Gemini)
   - ✅ End-to-end RAG workflow

2. **Custom Evaluation Pipeline (Fully LLM-based — `eval/eval_rag_updated.py`)**
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
   - ✅ Retrieved chunks logged to console per question (for transparency)
   - ✅ `build_llm()` dynamically reads `OLLAMA_BASE_URL` from `.env` (supports local + remote Colab tunnel)
   - ✅ `request_timeout=180.0` added to prevent hangs with large remote models

3. **Enhanced Retrieval System**
   - ✅ Cosine similarity scoring: `sim = 1.0 - float(dist)` (correct formula)
   - ✅ Hybrid search with over-fetching (top_k × 2)
   - ✅ Metadata filtering support
   - ✅ Deduplication by content hash
   - ✅ Enhanced scoring with heading overlap
   - ✅ Content length weighting
   - ✅ Score threshold filtering post-enhanced-scoring
   - ✅ Source citation formatting

4. **Memory-Safe Document Ingestion (`rebuild_vector_store.py`)**
   - ✅ Lightweight Docling converter (OCR off, table structure only)
   - ✅ Page-by-page PDF loading via `PdfReader` page count + `page_range` arg
   - ✅ Non-PDF files use default Docling converter
   - ✅ Single converter reused across all files

5. **Interactive Interfaces**
   - ✅ CLI chat interface (`chat.py`)
   - ✅ Jupyter notebook workflows
   - ✅ Environment-based configuration

### Debugging Achievements

- **Cosine Similarity Fix:** Fixed `sim = 1 - distance` → `sim = 1.0 - float(dist)` for ChromaDB cosine space
- **ChromaDB Cosine Space:** Added `"hnsw:space": "cosine"` to collection metadata for correct distance semantics
- **Enhanced Scoring Activation:** Enhanced score was previously set to `sim` directly; now calls `_enhanced_score()` properly and applies threshold _after_ enhanced scoring
- **Memory Issues Resolved:** `rebuild_vector_store.py` now uses a lightweight `DocumentConverter` with all heavy ML features disabled and processes PDFs one page at a time
- **Metadata Compatibility:** Resolved ChromaDB metadata type restrictions through flattening

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Document Ingestion                        │
│  DoclingLoader → Page-by-Page PDF Loading → Metadata         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Embedding Generation                        │
│  SentenceTransformer → Dense Embeddings (384-dim)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vector Store (ChromaDB)                     │
│  Cosine Space → Metadata Flattening → Persistent Indexing    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                         │
│  Query Embedding → Cosine Search → Enhanced Re-ranking       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Context Formatting                           │
│  Source Citations → Section Headings → Length Management      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Generation                           │
│  Prompt Engineering → Answer Generation → Response            │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
User Query → RAGRetriever.retrieve()
    ↓
EmbeddingManager.embed(query)
    ↓
VectorStore.collection.query()  [cosine space]
    ↓
RAGRetriever._process_and_rank()  [sim = 1.0 - dist, enhanced score]
    ↓
RAGRetriever.format_context_for_llm()  /  .format_context()  [eval alias]
    ↓
ask() → LLM.invoke(prompt)
    ↓
Formatted Answer with Citations
```

---

## Project Structure

```
inteli_docs_rag/
├── rag/                          # Core RAG package
│   ├── __init__.py              # Public API exports
│   ├── config.py                # Centralized configuration
│   └── pipeline.py              # Core RAG pipeline logic
├── eval/                         # Evaluation pipeline
│   ├── eval_rag_updated.py      # ✅ Current evaluation (LLM judges, 6 metrics)
│   ├── evaluate_rag.py          # Legacy evaluation script
│   ├── test_dataset.csv         # Evaluation questions (Question + Reference Answer)
│   └── eval_results_*.json      # Evaluation results
├── data/                         # Data directory
│   ├── pdf/                     # PDF documents
│   ├── text/                    # Text documents
│   └── vector_store/            # ChromaDB persistent storage
├── docs/                         # Documentation
│   ├── memory.md                # This file
│   └── 2_probs.excalidraw       # Architecture diagrams
├── notebook/                     # Jupyter notebooks
│   ├── 1_document.ipynb         # Initial document processing
│   ├── 2_.ipynb                 # Additional experiments
│   └── 3_docling.ipynb          # Docling integration & memory-safe loading
├── chat.py                       # Interactive CLI interface
├── rebuild_vector_store.py       # Memory-safe vector store rebuild script
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
└── .env                         # Actual environment variables (gitignored)
```

---

## Core Components

### 1. Configuration System (`rag/config.py`)

**Purpose:** Single source of truth for all tunable constants.

**Key Configuration Parameters:**

```python
# Paths
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "pdf_documents"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval Parameters
TOP_K = 5
MAX_CONTEXT_CHARS = 4000
SCORE_THRESHOLD = 0.25          # ← Raised from 0.0 to filter weak matches

# Document Ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = ["*.pdf", "*.ppt", "*.pptx", "*.doc", "*.docx", "*.txt"]

# LLM Backends
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
OLLAMA_MODEL = "qwen2.5:7b"            # Upgraded from 0.5b → 7b (Colab GPU)
OLLAMA_BASE_URL = "http://localhost:11434"   # Overridden by .env for remote Colab tunnel
OLLAMA_TEMPERATURE = 0.0
```

> **Important:** `SCORE_THRESHOLD` was changed from `0.0` to `0.25`. The threshold is now applied **after** `_enhanced_score()` is computed, not on the raw similarity.

**RAG Prompt Template:**
```python
RAG_PROMPT_TEMPLATE = """\
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
```

### 2. Embedding Manager (`rag/pipeline.py`)

**Purpose:** Loads SentenceTransformer model and generates dense embeddings.

**Class Structure:**

```python
class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Return (len(texts), dim) float32 ndarray."""
        return self.model.encode(texts, show_progress_bar=show_progress_bar)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Alias for embed() with progress bar (notebook-friendly)."""
        return self.embed(texts, show_progress_bar=True)
```

**Usage:**
```python
from rag import EmbeddingManager

embedder = EmbeddingManager()
embeddings = embedder.embed(["query text", "document text"])
# Returns: numpy array of shape (2, 384)
```

### 3. Vector Store (`rag/pipeline.py`)

**Purpose:** ChromaDB wrapper for persistent vector storage using cosine similarity space.

**Class Structure:**

```python
class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME,
                 persist_directory: str | Path = VECTOR_STORE_DIR) -> None:
        import chromadb
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Document embeddings for RAG",
                "hnsw:space": "cosine",      # ← Cosine distance space
            },
        )
```

> **Critical change:** The collection now explicitly uses `"hnsw:space": "cosine"`. This means ChromaDB returns cosine distances in the range `[0, 2]`. The correct similarity conversion is `sim = 1.0 - dist` (not `sim = -dist` as in a previous buggy version).

**Metadata Flattening Logic:**

```python
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
```

**Why This Matters:**
- ChromaDB only accepts simple types in metadata
- Docling generates complex nested metadata
- Flattening ensures compatibility while preserving information

### 4. RAG Retriever (`rag/pipeline.py`)

**Purpose:** Query vector store, re-rank results, and format context for LLM.

**Class Structure:**

```python
class RAGRetriever:
    def __init__(self, vector_store: VectorStore,
                 embedding_manager: EmbeddingManager) -> None: ...

    def retrieve(self, query: str, top_k: int = TOP_K,
                 score_threshold: float = SCORE_THRESHOLD,
                 filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]: ...

    def format_context_for_llm(self, retrieved_docs, max_context_length=MAX_CONTEXT_CHARS) -> str: ...

    def format_context(self, docs, max_chars=MAX_CONTEXT_CHARS) -> str:
        """Alias for format_context_for_llm() — used by eval pipeline."""
        return self.format_context_for_llm(docs, max_context_length=max_chars)
```

**Enhanced Ranking Logic (current correct version):**

```python
def _process_and_rank(self, results, query, top_k, score_threshold):
    for i, (doc_id, text, meta, dist) in enumerate(zip(ids, docs_raw, metas, distances)):
        # Correct cosine similarity: collection uses hnsw:space=cosine, dist in [0,2]
        sim = 1.0 - float(dist)

        # Deduplicate by content hash
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)

        # Compute enhanced score (heading overlap + length boost)
        enhanced_score = self._enhanced_score(
            document=text,
            query=query,
            similarity_score=sim,
            metadata=meta,
        )

        # Apply threshold AFTER enhanced scoring
        if enhanced_score < score_threshold:
            continue

        ranked.append({
            "id":               doc_id,
            "content":          text,
            "metadata":         meta,
            "similarity_score": sim,
            "enhanced_score":   enhanced_score,
            "distance":         dist,
            "rank":             i + 1,
            "similarity":       sim,   # eval-compat alias
        })

    ranked.sort(key=lambda x: x["enhanced_score"], reverse=True)
    return ranked[:top_k]
```

**Enhanced Scoring:**

```python
def _enhanced_score(self, document, query, similarity_score, metadata) -> float:
    score = similarity_score

    # Boost for heading/query word overlap
    headings = metadata.get("headings", "")
    if headings:
        q_terms = set(query.lower().split())
        h_terms = set(headings.lower().split())
        overlap = len(q_terms & h_terms)
        if overlap > 0:
            score += 0.1 * overlap

    # Boost for content length
    length = len(document)
    if length > 1000:
        score += 0.1
    elif length > 500:
        score += 0.05

    return min(score, 1.0)
```

**Context Formatting:**

```python
def format_context_for_llm(self, retrieved_docs, max_context_length=MAX_CONTEXT_CHARS) -> str:
    for doc in retrieved_docs:
        meta = doc["metadata"]
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

    return f"Context from {len(parts)} document(s):\n\n" + "\n\n---\n\n".join(parts)
```

### 5. High-Level RAG Function (`rag/pipeline.py`)

**Purpose:** One-call RAG entry point integrating retrieval, formatting, and generation.

```python
def ask(query: str, retriever: RAGRetriever, llm: Any,
        top_k: int = TOP_K, max_context_length: int = MAX_CONTEXT_CHARS,
        filter_metadata: Optional[Dict] = None) -> str:
    """High-level RAG call: retrieve → format context → generate answer."""
    docs = retriever.retrieve(query, top_k=top_k, filter_metadata=filter_metadata)
    if not docs:
        return "I don't have enough relevant information to answer this question."

    context = retriever.format_context_for_llm(docs, max_context_length)
    prompt  = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content
```

---

## RAG Pipeline

### Complete Workflow

1. **Document Ingestion**
   - `rebuild_vector_store.py` builds a lightweight Docling converter (OCR off, batch_size=1)
   - PDFs are loaded page-by-page using `PdfReader` to count pages + `page_range` kwarg
   - Non-PDF files (DOCX, TXT, PPTX) use the default Docling loader

2. **Embedding Generation**
   - SentenceTransformer model (all-MiniLM-L6-v2)
   - Generates 384-dimensional dense embeddings

3. **Vector Storage**
   - ChromaDB persistent storage with `hnsw:space: cosine`
   - Metadata flattening for compatibility
   - Unique document IDs with UUID

4. **Query Processing**
   - User query embedded using same model
   - Cosine similarity search in ChromaDB
   - Over-fetching (top_k * 2) for re-ranking

5. **Result Processing**
   - `sim = 1.0 - dist` (correct for cosine space)
   - Deduplication by content hash
   - Enhanced scoring (heading overlap + content length)
   - Threshold applied **after** enhanced scoring
   - Final top-k selection

6. **Context Formatting**
   - Source citation (filename, page, section)
   - Length management (max_context_chars)
   - Clear document separators

7. **LLM Generation**
   - Prompt engineering with instructions
   - Source citation requirements
   - Fallback for insufficient context

---

## Document Ingestion

### Memory-Safe PDF Loading (`rebuild_vector_store.py`)

**Problem solved:** `std::bad_alloc` errors from RapidOCR when processing large/complex PDFs.

**Solution:** A lightweight Docling `DocumentConverter` is built once with all heavy ML features disabled, then reused for every PDF. Each PDF is processed one page at a time.

```python
def build_light_pdf_converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions(
        force_backend_text=True,
        do_ocr=False,                      # OCR disabled — main memory saver
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,       # RAM-heavy, disabled
        do_chart_extraction=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=False,
        document_timeout=120,
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        queue_max_size=1,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )
```

**Page-by-page loading:**

```python
def load_pdf_documents(file_path: Path, converter: DocumentConverter) -> list:
    all_documents = []
    page_count = len(PdfReader(str(file_path)).pages)

    for page_no in range(1, page_count + 1):
        loader = DoclingLoader(
            str(file_path),
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            convert_kwargs={"page_range": (page_no, page_no)},
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = file_path.name
            doc.metadata["file_type"] = "pdf"
            doc.metadata["source_page_range"] = str(page_no)
        all_documents.extend(docs)
        gc.collect()

    return all_documents
```

**Key changes vs. old approach:**
- Old: `--max-pages-per-doc` CLI flag to cap pages (now removed)
- New: All pages processed, one at a time, with a memory-efficient converter
- Non-PDF files still use the default `DoclingLoader` (no custom converter needed)

### Docling Metadata Structure

```python
{
    'source': 'path/to/document.pdf',
    'dl_meta': {
        'schema_name': 'docling_core.transforms.chunker.DocMeta',
        'doc_items': [
            {
                'prov': [{'page_no': 1, 'bbox': {...}, 'charspan': [0, 187]}]
            }
        ],
        'headings': ['Document Title', 'Section Name'],
        'origin': {
            'mimetype': 'application/pdf',
            'filename': 'document.pdf'
        }
    }
}
```

---

## Embedding Generation

### SentenceTransformer Model

**Model:** `all-MiniLM-L6-v2`

**Specifications:**
- **Dimensions:** 384
- **Type:** Dense embeddings
- **Framework:** SentenceTransformers

**Batch Processing:**

```python
texts = ["document 1", "document 2", "document 3"]
embeddings = embedder.embed(texts, show_progress_bar=True)
# Shape: (3, 384)
```

---

## Vector Store Management

### ChromaDB Configuration

**Collection Name:** `pdf_documents`  
**Persist Directory:** `data/vector_store`  
**Distance Space:** `cosine` (hnsw:space = cosine)

> Distances returned by ChromaDB are in range [0, 2] for cosine. Use `sim = 1.0 - dist` to convert to similarity.

### Document Storage Schema

```python
{
    'id': 'doc_abc12345_0',
    'embedding': [0.1, 0.2, ...],   # 384-dimensional vector
    'metadata': {
        'filename': 'document.pdf',
        'file_type': 'pdf',
        'page_no': 1,
        'headings': 'Introduction',
        'doc_index': 0,
        'content_length': 500,
        'source_page_range': '1',   # Added by rebuild_vector_store.py for PDFs
        'dl_meta': '{...json...}'
    },
    'document': 'Document text content...'
}
```

---

## Retrieval Pipeline

### Enhanced Retrieval Process

**Step 1: Query Embedding**
```python
query_embedding = self.embedding_manager.embed([query])[0]
```

**Step 2: Vector Search (cosine)**
```python
results = self.vector_store.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=top_k * 2   # Over-fetch for re-ranking
)
```

**Step 3: Similarity Conversion**
```python
# hnsw:space=cosine → dist in [0,2], convert to similarity in [-1,1]
sim = 1.0 - float(dist)
```

**Step 4: Enhanced Scoring**
```python
enhanced_score = self._enhanced_score(document, query, sim, metadata)
if enhanced_score < score_threshold:   # Threshold applied here, not on raw sim
    continue
```

**Step 5: Final Ranking**
```python
ranked.sort(key=lambda x: x["enhanced_score"], reverse=True)
return ranked[:top_k]
```

### Similarity Score — History of Fixes

| Version | Formula | Status |
|---|---|---|
| Initial | `sim = 1 - distance` (on l2 space) | ❌ Incorrect — all filtered out |
| Fix 1 (commit a131c11) | `sim = float(-dist)` | ⚠️ Workaround — works for ranking but not semantically correct |
| Fix 2 (commit a131c11) | `sim = 1.0 - float(dist)` + `hnsw:space=cosine` | ✅ Correct — cosine space, dist in [0,2] |

### Metadata Filtering

```python
docs = retriever.retrieve(query, filter_metadata={"file_type": "pdf"})
docs = retriever.retrieve(query, filter_metadata={"filename": "specific.pdf"})
docs = retriever.retrieve(query, filter_metadata={"page_no": 1})
```

---

## LLM Integration

### Supported Backends

#### 1. Google Gemini

**Model:** `gemini-2.5-flash`

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=GEMINI_TEMPERATURE,
    api_key=os.getenv("GEMINI_API_KEY")
)
```

#### 2. Ollama (Local or Remote via Colab + Ngrok)

**Model:** `qwen2.5:7b` — upgraded from `qwen2.5:0.5b` for better evaluation quality.

**Running via Google Colab:**
Since `qwen2.5:7b` is too large for local CPU inference at a reasonable speed, Ollama is run on a **Google Colab GPU instance** and exposed to the local machine via an **Ngrok tunnel**.

**Setup pattern:**
1. In Colab: install Ollama, pull the model, run `ollama serve`.
2. In Colab: install `pyngrok`, create a TCP tunnel on port `11434`.
3. Copy the `ngrok` public URL (e.g., `https://xxxx.ngrok-free.app`) into your local `.env` as `OLLAMA_BASE_URL`.
4. The `build_llm()` function in `eval_rag_updated.py` reads this at runtime.

```python
# eval/eval_rag_updated.py — build_llm()
from langchain_ollama import ChatOllama

load_env()
target_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)  # Picks up Colab tunnel URL

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=target_url,
    temperature=0.0,
    request_timeout=180.0,   # Long timeout for remote model over tunnel
)
```

> **Important:** The `.env` file (gitignored) holds the real Ngrok URL. The `.env.example` file holds a placeholder. Never hardcode the Ngrok URL in code.

### LLM Factory Pattern (`chat.py`)

```python
def build_llm():
    backend = os.getenv("LLM_BACKEND", "gemini").lower()
    if backend == "ollama":
        return _build_ollama()
    return _build_gemini()
```

---

## Custom Evaluation Pipeline

> **Active file:** `eval/eval_rag_updated.py`  
> The legacy `eval/evaluate_rag.py` is retained but the updated version is the canonical evaluator.

### Design Philosophy

The updated pipeline uses **fully LLM-based evaluation** — no cosine similarity for retrieval relevance. Every metric is judged by the LLM (`qwen2.5:7b` via Ollama, running on Google Colab GPU and exposed via Ngrok tunnel).

The dataset only requires two columns: `Question` and `Reference Answer`.

### Evaluation Metrics

#### Metric 1: Retrieval Relevance (LLM-per-chunk)

**Purpose:** Judge whether each retrieved chunk actually contains evidence needed to answer the question — not just topic similarity.

**Judge prompt key criteria:**
- Relevant if: directly contains answer OR contains essential evidence
- Not relevant if: only same broad topic, semantically similar but unhelpful

**Derived metrics:**
- **Hit@K** — did any chunk in the top-K contain a relevant answer?
- **Precision@K** — fraction of retrieved chunks that are relevant
- **Mean Chunk Relevance** — mean LLM relevance score across all chunks

```python
def metric_retrieval_relevance(question, reference, retrieved_docs, llm):
    judgments = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        judgment = judge_retrieved_chunk(question, reference, doc["content"], llm)
        judgment["rank"] = rank
        judgment["distance"] = doc.get("distance")
        judgment["similarity"] = doc.get("similarity")
        judgments.append(judgment)

    hit_at_k       = float(any(j["relevant"] == 1 for j in judgments))
    precision_at_k = float(np.mean([j["relevant"] for j in judgments]))
    mean_relevance = float(np.mean([j["score"] for j in judgments]))

    return {"hit_at_k": ..., "precision_at_k": ..., "mean_relevance": ..., "chunk_judgments": judgments}
```

#### Metrics 2–4: Combined Answer Judge (single LLM call)

All three answer-level metrics are evaluated in a **single LLM call** for efficiency.

**Groundedness:** Are the factual claims in the answer supported by the retrieved context?
- High if: answer says "I don't know" (no unsupported claims)
- Low if: answer contradicts or fabricates beyond context

**Answer Correctness:** Does the generated answer convey the same facts as the reference?
- Semantic equivalence is acceptable
- Vague answers should not get full credit
- "I don't know" when reference has facts → score = 0

**Answer Relevance:** Does the generated answer directly address the question?
- Being on the same topic is not enough
- Must answer what was specifically asked

**Combined judge prompt response format:**
```json
{
    "groundedness": 0.0,
    "answer_correctness": 0.0,
    "answer_relevance": 0.0,
    "reasoning": {
        "groundedness": "...",
        "answer_correctness": "...",
        "answer_relevance": "..."
    }
}
```

### Evaluation Loop

```python
def run_evaluation():
    load_env()
    df = pd.read_csv(DATASET_PATH)   # Needs: Question, Reference Answer

    embedder  = EmbeddingManager()
    store     = VectorStore(collection_name=COLLECTION_NAME, persist_directory=VECTOR_STORE_DIR)
    retriever = RAGRetriever(store, embedder)
    llm       = build_llm()   # Ollama — local or remote Colab tunnel

    for idx, row in df.iterrows():
        answer, context, docs = generate_answer(question, retriever, llm)

        # Log each retrieved chunk to console (added for transparency)
        for i, doc in enumerate(docs):
            logger.info("  Context [%d] (sim: %.4f): %s...", i+1, doc.get("similarity") or 0.0, doc["content"][:100])

        retrieval_metrics = metric_retrieval_relevance(question, reference, docs, llm)
        answer_metrics    = metric_llm_judges(question, reference, context, answer, llm)

        results.append({...})

        # ✅ Incremental save after every question — prevents data loss on crash/interrupt
        log_path.write_text(json.dumps({"partial_results": results}, indent=2), encoding="utf-8")

    # Final full save with aggregation: mean/std/min/max for each metric
    summary = {
        "aggregate": {
            "retrieval_hit_at_k":              aggregate_metric(hit_values),
            "retrieval_precision_at_k":        aggregate_metric(precision_values),
            "retrieval_mean_chunk_relevance":  aggregate_metric(relevance_values),
            "groundedness":                    aggregate_metric(groundedness_values),
            "answer_correctness":              aggregate_metric(correctness_values),
            "answer_relevance":                aggregate_metric(answer_relevance_values),
        },
        "per_question_results": results,
    }
```

### Robustness Features

**LLM Retry with Exponential Backoff:**
```python
def _call_llm_with_retry(llm, prompt, retries=3) -> str:
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # 1s, 2s, 4s
    return ""
```

**JSON Parsing with Fence Stripping:**
```python
def clean_json_response(raw: str) -> str:
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end+1] if start != -1 and end != -1 else raw
```

**Score Clamping:**
```python
def clamp_score(value) -> float:
    return max(0.0, min(1.0, float(value)))
```

### Output Files

- **JSON:** `eval/eval_results_<timestamp>.json` — full per-question results with chunk judgments and reasoning
- **CSV:** `eval/eval_summary_<timestamp>.csv` — flattened per-question scores for spreadsheet analysis

**Summary table format printed to console:**
```
======================================================================
EVALUATION SUMMARY
======================================================================
Metric                                   Mean     Std     Min     Max
----------------------------------------------------------------------
retrieval_hit_at_k                     0.8000  0.4000  0.0000  1.0000
retrieval_precision_at_k               0.5600  0.2800  0.2000  0.8000
retrieval_mean_chunk_relevance         0.6100  0.2200  0.3000  0.9000
groundedness                           0.7800  0.1500  0.5000  0.9500
answer_correctness                     0.7100  0.1800  0.4000  0.9200
answer_relevance                       0.8900  0.1000  0.7000  1.0000
======================================================================
```

---

## Known Issues & Debugging

### Issue 1: Metadata Type Error

**Error:**
```
ValueError: Expected metadata value to be a str, int, float, bool, or None, got {...}
```

**Cause:** Docling generates complex nested metadata; ChromaDB only accepts simple types.

**Solution:** `VectorStore.add_documents()` flattens all metadata before insertion (see [Vector Store section](#3-vector-store-ragpipelinepy)).

### Issue 2: Memory Allocation Errors (Resolved)

**Error:**
```
Stage preprocess failed for run 1, pages [5]: std::bad_alloc
```

**Old mitigation:** `--max-pages-per-doc` CLI flag (now removed).

**Current solution:** `build_light_pdf_converter()` in `rebuild_vector_store.py` disables all heavy ML features (OCR, picture description, formula enrichment, etc.) and loads PDFs one page at a time.

### Issue 3: Similarity Score Bug (Resolved)

**Timeline:**

| Stage | Code | Problem |
|---|---|---|
| Original | `sim = 1 - distance` on default l2 space | All results filtered out (`sim` was ~0 for l2 distances) |
| Fix attempt | `sim = float(-dist)` | Worked for ranking but semantically wrong |
| **Current (correct)** | `sim = 1.0 - float(dist)` with `hnsw:space=cosine` | Correct: cosine dist ∈ [0,2], sim ∈ [-1,1] |

**Root cause:** ChromaDB's default space is l2; distances are large unbounded values. With the `hnsw:space=cosine` collection setting, distances are cosine distances ∈ [0,2] and `1 - dist` gives correct cosine similarity.

### Issue 4: Enhanced Score Not Applied (Resolved — commit a131c11)

**Old buggy code:**
```python
ranked.append({
    ...
    "enhanced_score": sim,   # ← Was just sim, _enhanced_score() never called
})
```

**Fixed code:**
```python
enhanced_score = self._enhanced_score(document=text, query=query,
                                       similarity_score=sim, metadata=meta)
if enhanced_score < score_threshold:
    continue
ranked.append({
    ...
    "enhanced_score": enhanced_score,   # ← Now uses actual enhanced score
})
```

### Issue 5: Score Threshold Applied at Wrong Stage

The threshold was previously checked against raw `sim`; now it's applied against the fully-computed `enhanced_score`. This matters because heading overlap can push borderline chunks over the threshold.

### Debugging Checklist for Empty Retrieval

1. Check collection has documents: `store.collection.count()`
2. Check raw distances: print `distances` from `results`
3. Verify `hnsw:space` is `cosine`: `store.collection.metadata`
4. Temporarily set `SCORE_THRESHOLD = 0.0` to confirm retrieval works
5. Print `sim = 1.0 - dist` for each result — should be ~0.3–0.9 for good matches

---

## Production Considerations

### Scalability

**Document Processing:**
- Memory allocation errors now mitigated by lightweight converter + page-by-page loading
- Single-threaded; parallelize with worker pool for large document sets

**Recommendations:**
1. Queue-based processing (Celery, RabbitMQ) for async ingestion
2. Horizontal scaling with multiple worker processes
3. Consider cloud OCR (AWS Textract, Google Document AI) for scanned documents

**Vector Store:**
- ChromaDB suitable for small-medium datasets
- Alternatives for large scale: Qdrant, Weaviate, Pinecone

### Performance Optimization

- Batch embedding with progress bars for large document sets
- GPU acceleration for SentenceTransformer
- Cache frequent query embeddings
- Tune `HNSW` parameters for faster ANN search

### Error Handling

```python
try:
    documents = load_documents(file_path, pdf_converter)
except Exception as exc:
    logger.exception("Error processing %s: %s", file_path, exc)
finally:
    gc.collect()
```

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
# Edit .env and set GEMINI_API_KEY
```

**Start Ollama locally (small model):**
```bash
ollama serve
ollama pull qwen2.5:0.5b   # Lightweight, fast, lower quality
```

**Run via Google Colab (recommended — 7B model):**
1. Open a Colab notebook with GPU runtime (T4 is sufficient).
2. Install and start Ollama, then pull the model:
   ```bash
   !curl -fsSL https://ollama.com/install.sh | sh
   !ollama serve &
   !ollama pull qwen2.5:7b
   ```
3. Create a public Ngrok tunnel:
   ```python
   from pyngrok import ngrok
   tunnel = ngrok.connect(11434, "tcp")
   print(tunnel.public_url)  # e.g. tcp://xxxx.ngrok-free.app:PORT
   ```
4. Set `OLLAMA_BASE_URL` in your local `.env` to the printed Ngrok URL.
5. Run the eval script locally — it will route all LLM calls through Colab.

### Rebuild Vector Store

```bash
# Wipe existing store and rebuild from data/
python rebuild_vector_store.py

# Keep existing documents and add more
python rebuild_vector_store.py --keep-existing

# Use a custom data directory
python rebuild_vector_store.py --data-dir /path/to/docs
```

> **Note:** The `--max-pages-per-doc` flag has been removed. The script now processes all pages via the memory-safe page-by-page loader.

### Interactive Chat

```bash
# Default: Gemini
python chat.py

# Use Ollama
LLM_BACKEND=ollama python chat.py
```

**Chat Commands:**
- Type question → Enter
- `exit` / `quit` / `q` → leave

### Running Evaluation

```bash
# Run the updated evaluation pipeline
python eval/eval_rag_updated.py
```

The CSV dataset at `eval/test_dataset.csv` must have columns: `Question`, `Reference Answer`.

**Output:**
- `eval/eval_results_<timestamp>.json` — detailed results (also saved incrementally after each question)
- `eval/eval_summary_<timestamp>.csv` — per-question scores

> **Crash Safety:** The JSON file is written after every single question. If the script is interrupted mid-run, you will not lose data already processed.

### Programmatic Usage

```python
from rag import EmbeddingManager, VectorStore, RAGRetriever, ask
from langchain_google_genai import ChatGoogleGenerativeAI
import os

embedder  = EmbeddingManager()
store     = VectorStore()
retriever = RAGRetriever(store, embedder)
llm       = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                   api_key=os.getenv("GEMINI_API_KEY"))

answer = ask("What is the main topic?", retriever, llm)
print(answer)
```

---

## Development Notes

### Code Organization Principles

- **Single Source of Truth:** All configuration in `rag/config.py`
- **Lazy Imports:** Heavy dependencies imported inside functions/classes
- **Eval Compatibility:** `format_context()` alias keeps eval pipeline decoupled from method renames
- **Modular Architecture:** `chat.py`, `eval/`, and `rebuild_vector_store.py` all import from `rag` package

### API Reference

**rag.EmbeddingManager**
```python
class EmbeddingManager:
    def embed(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray
    def generate_embeddings(self, texts: List[str]) -> np.ndarray   # notebook alias
```

**rag.VectorStore**
```python
class VectorStore:
    def __init__(self, collection_name: str, persist_directory: Path) -> None
    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None
```

**rag.RAGRetriever**
```python
class RAGRetriever:
    def retrieve(self, query: str, top_k: int, score_threshold: float,
                 filter_metadata: Optional[Dict]) -> List[Dict[str, Any]]
    def format_context_for_llm(self, retrieved_docs, max_context_length: int) -> str
    def format_context(self, docs, max_chars: int) -> str    # eval alias
```

**rag.ask**
```python
def ask(query: str, retriever: RAGRetriever, llm: Any,
        top_k: int = TOP_K, max_context_length: int = MAX_CONTEXT_CHARS,
        filter_metadata: Optional[Dict] = None) -> str
```

### Environment Variables

```
GEMINI_API_KEY      # Required for Gemini backend
LLM_BACKEND         # "gemini" (default) or "ollama"
```

### Future Enhancements

1. **Hybrid Search:** Combine semantic and keyword (BM25) search
2. **Query Expansion / Multi-Query:** Retrieve with multiple query reformulations
3. **Streaming Responses:** Real-time LLM response streaming in chat.py
4. **Web Interface:** React/FastAPI-based chat UI
5. **API Endpoints:** REST API for integration
6. **Advanced Metrics:** Faithfulness, answer consistency
7. **A/B Testing:** Compare retrieval strategies
8. **User Feedback Loop:** Incorporate user ratings

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'rag'` | Not running from project root | `cd inteli_docs_rag; set PYTHONPATH=%CD%` |
| `sqlite3.OperationalError: database is locked` | Multiple processes accessing ChromaDB | Kill zombie processes; use one process at a time |
| `ConnectionError: Failed to connect to Ollama` | Ollama not running | `ollama serve` |
| `RateLimitError: API rate limit exceeded` | Gemini API quota hit | Use Ollama for eval; implement backoff |
| `std::bad_alloc` from Docling | Heavy ML features + large PDF | Use `rebuild_vector_store.py` (light converter, page-by-page) |
| All retrieved docs empty | Wrong distance formula or threshold too high | Check `hnsw:space=cosine`; set `SCORE_THRESHOLD=0.0` to debug |
| Eval LLM times out / hangs | Model on Colab too slow or tunnel dropped | Increase `request_timeout` in `build_llm()`; restart Colab + Ngrok tunnel |
| Eval results not saved after crash | Script crashed before final save block | Results up to last completed question are in the JSON file (incremental save) |

---

**Document Version:** 2.1  
**Last Updated:** August 3, 2026  
**Covers Commits:** `2e41697` → `a131c11` → `e107f97` → `3a55bea` → (Aug 03 changes)  
**Status:** Updated — Ollama upgraded to qwen2.5:7b via Google Colab + Ngrok, eval pipeline hardened (incremental saves, chunk logging, dynamic URL, request timeout), pipeline.py cleaned up
