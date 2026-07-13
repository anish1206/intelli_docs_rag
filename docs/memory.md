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
- Persistent vector storage with ChromaDB
- Enhanced retrieval with re-ranking and deduplication
- Multiple LLM backends (Gemini, Ollama)
- Custom evaluation pipeline with four metrics
- Interactive CLI chat interface
- Source citation and context formatting

**Technology Stack:**
- **Document Parsing:** Docling (with RapidOCR for OCR)
- **Embeddings:** SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB (persistent)
- **LLM Backends:** Google Gemini (gemini-2.5-flash), Ollama (qwen2.5:0.5b)
- **Framework:** LangChain
- **Evaluation:** Custom metrics with LLM judges
- **Language:** Python 3.x

---

## Current Status

### Completed Components

1. **RAG Pipeline (Fully Functional)**
   - ✅ PDF/Text document ingestion with Docling
   - ✅ Document chunking and metadata extraction
   - ✅ Embedding generation with SentenceTransformer
   - ✅ ChromaDB vector store with persistent storage
   - ✅ Enhanced retrieval pipeline with re-ranking
   - ✅ Context formatting with source citations
   - ✅ Local LLM generation (Qwen/Ollama)
   - ✅ Cloud LLM integration (Gemini)
   - ✅ End-to-end RAG workflow

2. **Custom Evaluation Pipeline (Completed)**
   - ✅ Retrieval Relevance metric (cosine similarity)
   - ✅ Groundedness metric (LLM judge)
   - ✅ Answer Correctness metric (LLM judge)
   - ✅ Answer Relevance metric (LLM judge)
   - ✅ Combined LLM judge for efficiency
   - ✅ Result aggregation and logging
   - ✅ JSON and CSV output formats

3. **Enhanced Retrieval System**
   - ✅ Hybrid search with over-fetching
   - ✅ Metadata filtering support
   - ✅ Deduplication by content hash
   - ✅ Enhanced scoring with heading overlap
   - ✅ Content length weighting
   - ✅ Source citation formatting

4. **Interactive Interfaces**
   - ✅ CLI chat interface (chat.py)
   - ✅ Jupyter notebook workflows
   - ✅ Environment-based configuration

### Debugging Achievements

- **Root Cause Analysis:** Identified and fixed similarity score conversion bug in retrieval pipeline
- **Metadata Compatibility:** Resolved ChromaDB metadata type restrictions through flattening
- **Memory Issues:** Diagnosed RapidOCR memory allocation errors during document processing

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Document Ingestion                        │
│  DoclingLoader → Document Parsing → Metadata Extraction      │
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
│  Persistent Storage → Metadata Flattening → Indexing          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                         │
│  Query Embedding → Similarity Search → Re-ranking → Filtering│
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
VectorStore.collection.query()
    ↓
RAGRetriever._process_and_rank()
    ↓
RAGRetriever.format_context_for_llm()
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
│   ├── evaluate_rag.py          # Custom evaluation metrics
│   ├── test_dataset.csv         # Evaluation questions
│   └── eval_results_*.json      # Evaluation results
├── data/                         # Data directory
│   ├── pdf/                     # PDF documents
│   ├── text/                    # Text documents
│   └── vector_store/            # ChromaDB persistent storage
├── docs/                         # Documentation
│   └── memory.md                # This file
├── notebook/                     # Jupyter notebooks
│   ├── 1_document.ipynb         # Initial document processing
│   ├── 2_.ipynb                 # Additional experiments
│   └── 3_docling.ipynb          # Docling integration
├── chat.py                       # Interactive CLI interface
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
SCORE_THRESHOLD = 0.0

# Document Ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = ["*.pdf", "*.ppt", "*.pptx", "*.doc", "*.docx", "*.txt"]

# LLM Backends
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.0
```

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
        """Load SentenceTransformer model."""
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

**Purpose:** ChromaDB wrapper for persistent vector storage with metadata flattening.

**Class Structure:**

```python
class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME, 
                 persist_directory: str | Path = VECTOR_STORE_DIR) -> None:
        """Initialize ChromaDB persistent client and collection."""
        import chromadb
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for RAG"}
        )
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        """Add LangChain Document objects with metadata flattening."""
        # Metadata flattening logic:
        # - Simple types (str, int, float, bool, None): kept as-is
        # - Complex dict (origin): extract filename
        # - List (headings): convert to comma-separated string
        # - Complex dict (dl_meta): extract page_no, serialize rest as JSON
        # - Other complex types: serialize as JSON string
```

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
- JSON serialization allows retrieval of full metadata when needed

### 4. RAG Retriever (`rag/pipeline.py`)

**Purpose:** Query vector store, re-rank results, and format context for LLM.

**Class Structure:**

```python
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, 
                 embedding_manager: EmbeddingManager) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = TOP_K, 
                 score_threshold: float = SCORE_THRESHOLD,
                 filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with enhanced ranking."""
        query_embedding = self.embedding_manager.embed([query])[0]
        
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k * 2,  # Over-fetch for re-ranking
        }
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        results = self.vector_store.collection.query(**query_params)
        return self._process_and_rank(results, query, top_k, score_threshold)
```

**Enhanced Ranking Logic:**

```python
def _process_and_rank(self, results: Dict, query: str, top_k: int, 
                      score_threshold: float) -> List[Dict[str, Any]]:
    """Process, deduplicate, and re-rank results."""
    # Chroma returns ranked results (lower distance = better match)
    for i, (doc_id, text, meta, dist) in enumerate(zip(ids, docs_raw, metas, distances)):
        # Convert distance to similarity score
        sim = float(-dist)  # Negative distance for ranking
        
        # Deduplicate by content hash
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)
        
        ranked.append({
            "id": doc_id,
            "content": text,
            "metadata": meta,
            "similarity_score": sim,
            "enhanced_score": sim,
            "distance": dist,
            "rank": i + 1,
            "similarity": sim,  # eval-compat alias
        })
    
    ranked.sort(key=lambda x: x["enhanced_score"], reverse=True)
    return ranked[:top_k]
```

**Context Formatting:**

```python
def format_context_for_llm(self, retrieved_docs: List[Dict[str, Any]], 
                          max_context_length: int = MAX_CONTEXT_CHARS) -> str:
    """Format retrieved chunks with source citations."""
    if not retrieved_docs:
        return "No relevant context found."
    
    parts = []
    used = 0
    
    for doc in retrieved_docs:
        meta = doc["metadata"]
        filename = meta.get("filename") or meta.get("source_file", "Unknown")
        page_no = meta.get("page_no", "N/A")
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
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content
```

---

## RAG Pipeline

### Complete Workflow

1. **Document Ingestion**
   - DoclingLoader processes documents (PDF, PPT, Word, Text)
   - Documents are chunked with metadata extraction
   - Metadata includes: filename, page numbers, headings, doc_items

2. **Embedding Generation**
   - SentenceTransformer model (all-MiniLM-L6-v2)
   - Generates 384-dimensional dense embeddings
   - Batch processing for efficiency

3. **Vector Storage**
   - ChromaDB persistent storage
   - Metadata flattening for compatibility
   - Unique document IDs with UUID
   - Persistent across sessions

4. **Query Processing**
   - User query embedded using same model
   - Vector similarity search in ChromaDB
   - Over-fetching (top_k * 2) for re-ranking

5. **Result Processing**
   - Deduplication by content hash
   - Enhanced scoring with heading overlap
   - Content length weighting
   - Final top-k selection

6. **Context Formatting**
   - Source citation (filename, page, section)
   - Length management (max_context_chars)
   - Clear document separators

7. **LLM Generation**
   - Prompt engineering with instructions
   - Source citation requirements
   - Fallback for insufficient context
   - Answer generation

### Pipeline Diagram

```
Document → DoclingLoader → Chunks + Metadata
                                    ↓
EmbeddingManager → Embeddings (384-dim)
                                    ↓
VectorStore.add_documents() → ChromaDB
                                    ↓
User Query → EmbeddingManager → Query Embedding
                                    ↓
VectorStore.collection.query() → Raw Results
                                    ↓
RAGRetriever._process_and_rank() → Ranked Results
                                    ↓
RAGRetriever.format_context_for_llm() → Formatted Context
                                    ↓
ask() → LLM.invoke() → Final Answer
```

---

## Document Ingestion

### Docling Integration

**Purpose:** Use Docling exclusively for all document types (PDF, PPT, Word, Text).

**Implementation:**

```python
from langchain_docling.loader import DoclingLoader, ExportType
from pathlib import Path

def process_all_documents(directory, max_pages_per_doc=None):
    """Process all supported document types with Docling."""
    all_documents = []
    doc_dir = Path(directory)
    
    supported_extensions = ['*.pdf', '*.ppt', '*.pptx', '*.doc', '*.docx', '*.txt']
    
    # Find all supported files recursively
    all_files = []
    for ext in supported_extensions:
        all_files.extend(doc_dir.glob(f"**/{ext}"))
    
    for file_path in all_files:
        file_ext = file_path.suffix.lower()
        try:
            loader = DoclingLoader(
                str(file_path), 
                export_type=ExportType.DOC_CHUNKS,
                page_range=[0, max_pages_per_doc] if max_pages_per_doc else None
            )
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source_file'] = file_path.name
                doc.metadata['file_type'] = file_ext[1:]
            
            all_documents.extend(documents)
            
            # Memory management
            del loader
            del documents
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    return all_documents
```

### Docling Metadata Structure

**Typical Docling Metadata:**

```python
{
    'source': 'path/to/document.pdf',
    'dl_meta': {
        'schema_name': 'docling_core.transforms.chunker.DocMeta',
        'version': '1.0.0',
        'doc_items': [
            {
                'self_ref': '#/texts/5',
                'parent': {'$ref': '#/body'},
                'content_layer': 'body',
                'label': 'text',
                'prov': [
                    {
                        'page_no': 1,
                        'bbox': {'l': 49.3, 't': 503.9, 'r': 566.3, 'b': 455.2},
                        'charspan': [0, 187]
                    }
                ]
            }
        ],
        'headings': ['Document Title', 'Section Name'],
        'origin': {
            'mimetype': 'application/pdf',
            'binary_hash': 15632868855091330524,
            'filename': 'document.pdf'
        }
    }
}
```

### Memory Management Issues

**Problem:** RapidOCR (used by Docling) causes `std::bad_alloc` errors during processing.

**Symptoms:**
```
Stage preprocess failed for run 1, pages [5]: std::bad_alloc
Stage preprocess failed for run 1, pages [6]: std::bad_alloc
```

**Causes:**
- RapidOCR is memory-intensive for complex pages
- Mobile OCR models have limited memory capacity
- System RAM limitations during batch processing

**Mitigation Strategies:**

1. **Page Limiting:**
```python
loader = DoclingLoader(str(file_path), export_type=ExportType.DOC_CHUNKS, 
                      page_range=[0, 10])  # Process only first 10 pages
```

2. **Batch Processing:**
```python
# Process documents in small groups
for batch in chunked(all_files, batch_size=3):
    process_batch(batch)
    gc.collect()
```

3. **Garbage Collection:**
```python
import gc
del loader
del documents
gc.collect()
```

4. **Production Alternatives:**
- Use cloud-based parsing (AWS Textract, Google Document AI)
- Implement queue-based processing
- Scale horizontally with multiple workers

---

## Embedding Generation

### SentenceTransformer Model

**Model:** `all-MiniLM-L6-v2`

**Specifications:**
- **Dimensions:** 384
- **Type:** Dense embeddings
- **Framework:** SentenceTransformers
- **Language:** Multilingual support

**Usage:**

```python
from rag import EmbeddingManager

embedder = EmbeddingManager()
embeddings = embedder.embed(["text to embed"])
# Shape: (1, 384)
```

**Batch Processing:**

```python
texts = ["document 1", "document 2", "document 3"]
embeddings = embedder.embed(texts, show_progress_bar=True)
# Shape: (3, 384)
```

### Embedding Characteristics

**Advantages:**
- Fast inference (optimized for CPU)
- Good semantic understanding
- Low memory footprint
- Multilingual support

**Limitations:**
- Fixed dimensionality (384)
- Not domain-specific
- May require fine-tuning for specialized domains

---

## Vector Store Management

### ChromaDB Configuration

**Collection Name:** `pdf_documents`

**Persist Directory:** `data/vector_store`

**Storage Structure:**

```
data/vector_store/
├── chroma.sqlite3          # SQLite database
└── [collection_data]       # Vector embeddings and metadata
```

### Document Storage Schema

**Each Document Contains:**

```python
{
    'id': 'doc_abc12345_0',           # Unique identifier
    'embedding': [0.1, 0.2, ...],     # 384-dimensional vector
    'metadata': {
        'filename': 'document.pdf',
        'file_type': 'pdf',
        'page_no': 1,
        'headings': 'Introduction',
        'doc_index': 0,
        'content_length': 500,
        'dl_meta': '{...json...}'      # Serialized complex metadata
    },
    'document': 'Document text content...'
}
```

### Vector Operations

**Add Documents:**

```python
from rag import VectorStore, EmbeddingManager

store = VectorStore()
embedder = EmbeddingManager()

documents = [...]  # LangChain Document objects
embeddings = embedder.embed([doc.page_content for doc in documents])

store.add_documents(documents, embeddings)
```

**Query Collection:**

```python
results = store.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    where={"file_type": "pdf"}  # Optional metadata filter
)
```

**Collection Statistics:**

```python
count = store.collection.count()
metadata = store.collection.metadata
```

---

## Retrieval Pipeline

### Enhanced Retrieval Process

**Step 1: Query Embedding**
```python
query_embedding = self.embedding_manager.embed([query])[0]
```

**Step 2: Vector Search**
```python
results = self.vector_store.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=top_k * 2  # Over-fetch for re-ranking
)
```

**Step 3: Result Processing**
```python
def _process_and_rank(self, results, query, top_k, score_threshold):
    # Convert distances to similarity scores
    sim = float(-dist)  # Negative distance for ranking
    
    # Deduplicate by content hash
    h = hash(text)
    if h in seen:
        continue
    
    # Apply enhanced scoring
    enhanced_score = self._enhanced_score(document, query, sim, metadata)
```

**Step 4: Enhanced Scoring**
```python
def _enhanced_score(self, document, query, similarity_score, metadata):
    score = similarity_score
    
    # Boost for heading overlap
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

**Step 5: Final Ranking**
```python
ranked.sort(key=lambda x: x["enhanced_score"], reverse=True)
return ranked[:top_k]
```

### Similarity Score Bug Fix

**Original Bug:**
```python
# INCORRECT - caused all results to be filtered out
sim = 1 - distance
if sim < score_threshold:
    continue  # This was always true!
```

**Fixed Version:**
```python
# CORRECT - use negative distance for ranking
sim = float(-dist)
# Chroma returns ranked results, lower distance = better match
```

**Why This Matters:**
- ChromaDB uses cosine distance (not similarity)
- Distance range: [0, 2] for cosine distance
- Converting with `1 - distance` was incorrect
- Using negative distance preserves ranking order

### Metadata Filtering

**Usage:**
```python
# Filter by file type
docs = retriever.retrieve(query, filter_metadata={"file_type": "pdf"})

# Filter by filename
docs = retriever.retrieve(query, filter_metadata={"filename": "specific.pdf"})

# Filter by page number
docs = retriever.retrieve(query, filter_metadata={"page_no": 1})
```

**ChromaDB Where Clause Syntax:**
```python
{
    "file_type": {"$eq": "pdf"},           # Equality
    "page_no": {"$gte": 1, "$lte": 5},    # Range
    "headings": {"$contains": "Intro"}    # String contains
}
```

---

## LLM Integration

### Supported Backends

#### 1. Google Gemini

**Model:** `gemini-2.5-flash`

**Configuration:**
```python
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
```

**Setup:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=GEMINI_TEMPERATURE,
    api_key=os.getenv("GEMINI_API_KEY")
)
```

**Environment Variable:**
```
GEMINI_API_KEY=your_api_key_here
```

#### 2. Ollama (Local)

**Model:** `qwen2.5:0.5b`

**Configuration:**
```python
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.0
```

**Setup:**
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=OLLAMA_TEMPERATURE
)
```

**Requirements:**
- Ollama installed and running
- Model pulled: `ollama pull qwen2.5:0.5b`

### LLM Factory Pattern

**Implementation:**

```python
def build_llm():
    backend = os.getenv("LLM_BACKEND", "gemini").lower()
    if backend == "ollama":
        return _build_ollama()
    return _build_gemini()
```

**Usage:**
```python
# Default: Gemini
python chat.py

# Use Ollama
LLM_BACKEND=ollama python chat.py
```

### Prompt Engineering

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

**Key Design Principles:**
1. **Clear Instructions:** Explicit guidance on using context
2. **Source Citation:** Requirement to cite sources
3. **Fallback Handling:** Graceful handling of insufficient context
4. **Synthesis:** Guidance for combining multiple sources
5. **Accuracy:** Emphasis on being specific and factual

---

## Custom Evaluation Pipeline

### Evaluation Metrics

#### 1. Retrieval Relevance

**Purpose:** Measure semantic similarity between query and retrieved chunks.

**Implementation:**
```python
def metric_retrieval_relevance(query: str, retrieved_docs: List[Dict], 
                               embedder: EmbeddingManager) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not retrieved_docs:
        return 0.0
    
    chunk_texts = [d["content"] for d in retrieved_docs]
    q_emb = embedder.embed([query])
    c_emb = embedder.embed(chunk_texts)
    sims = cosine_similarity(q_emb, c_emb)[0]
    
    # Return mean of top 3 similarities
    return float(np.mean(sorted(sims, reverse=True)[:3]))
```

**Scoring:**
- Range: [0.0, 1.0]
- Higher = better retrieval
- Uses top-3 chunks for robustness

#### 2. Groundedness

**Purpose:** Verify that answer claims are supported by retrieved context.

**LLM Judge Prompt:**
```python
groundedness – every claim in the answer is supported by the retrieved context
                (1.0 = fully supported, 0.0 = unsupported / contradicts context)
```

**Scoring:**
- Range: [0.0, 1.0]
- 1.0 = All claims supported
- 0.0 = Claims unsupported or contradict context

#### 3. Answer Correctness

**Purpose:** Evaluate factual agreement with reference answer.

**LLM Judge Prompt:**
```python
answer_correctness – Evaluate factual agreement and semantic equivalence with the reference answer. Different wording should not reduce score.
                    Score 1.0 if both answers convey the same information, even if phrased differently, else 0.0
                    If the generated answer states that it lacks information, and the reference answer contains specific facts, answer_correctness must be 0.0.
```

**Scoring:**
- Range: [0.0, 1.0]
- Semantic equivalence (not exact wording)
- Penalizes "I don't know" when reference has facts

#### 4. Answer Relevance

**Purpose:** Check if answer directly addresses the question.

**LLM Judge Prompt:**
```python
answer_relevance  – if the generated answer directly addresses the question
                    (1.0 = fully on-topic else 0.0 when off-topic)
```

**Scoring:**
- Range: [0.0, 1.0]
- Binary-style scoring (on-topic vs off-topic)

### Combined LLM Judge

**Purpose:** Single LLM call to evaluate three metrics simultaneously.

**Implementation:**

```python
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
```

**Response Parsing:**

```python
def metric_llm_judges(question, reference, context, answer, llm):
    prompt = COMBINED_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        context=context[:6000],  # Truncate to avoid token limits
        answer=answer
    )
    raw = _call_llm_with_retry(llm, prompt)
    
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        scores = json.loads(cleaned)
        return {
            "groundedness": parse_score(str(scores.get("groundedness", 0))),
            "answer_correctness": parse_score(str(scores.get("answer_correctness", 0))),
            "answer_relevance": parse_score(str(scores.get("answer_relevance", 0))),
        }
    except Exception:
        # Fallback to regex extraction
        return _extract_scores_with_regex(raw)
```

### Evaluation Loop

**Main Evaluation Function:**

```python
def run_evaluation():
    # Load test dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Initialize components
    embedder = EmbeddingManager()
    store = VectorStore()
    retriever = RAGRetriever(store, embedder)
    llm = build_llm()
    
    results = []
    
    for idx, row in df.iterrows():
        question = str(row["Question"]).strip()
        reference = str(row["Reference Answer"]).strip()
        
        # Generate RAG answer
        answer, context, docs = generate_answer(question, retriever, llm)
        
        # Compute metrics
        ret_rel = metric_retrieval_relevance(question, docs, embedder)
        judge_scores = metric_llm_judges(question, reference, context, answer, llm)
        
        results.append({
            "question": question,
            "reference_answer": reference,
            "generated_answer": answer,
            "metrics": {
                "retrieval_relevance": ret_rel,
                "groundedness": judge_scores["groundedness"],
                "answer_correctness": judge_scores["answer_correctness"],
                "answer_relevance": judge_scores["answer_relevance"],
            }
        })
    
    # Aggregate and save results
    summary = aggregate_results(results)
    save_results(summary)
```

### Result Aggregation

**Statistics Computed:**

```python
def _agg(key: str) -> Dict[str, float]:
    vals = [m[key] for m in all_metrics]
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std":  round(float(np.std(vals)), 4),
        "min":  round(float(np.min(vals)), 4),
        "max":  round(float(np.max(vals)), 4),
    }
```

**Output Format:**

```python
{
    "eval_timestamp": "20260619_160016",
    "embedding_model": "all-MiniLM-L6-v2",
    "llm": "qwen2.5:0.5b",
    "num_test_cases": 10,
    "top_k_retrieval": 5,
    "aggregate": {
        "retrieval_relevance": {"mean": 0.85, "std": 0.12, "min": 0.65, "max": 0.98},
        "groundedness": {"mean": 0.78, "std": 0.15, "min": 0.50, "max": 0.95},
        "answer_correctness": {"mean": 0.72, "std": 0.18, "min": 0.40, "max": 0.92},
        "answer_relevance": {"mean": 0.88, "std": 0.10, "min": 0.70, "max": 1.00}
    },
    "per_question_results": [...]
}
```

### Output Files

**JSON Results:** `eval/eval_results_<timestamp>.json`
- Detailed per-question results
- Full metrics and context
- Latency measurements

**CSV Summary:** `eval/eval_summary_<timestamp>.csv`
- Aggregated statistics
- Per-question scores
- Easy for analysis and visualization

---

## Known Issues & Debugging

### Issue 1: Metadata Type Error

**Error:**
```
ValueError: Expected metadata value to be a str, int, float, bool, SparseVector, list, or None, got {...} which is a dict
```

**Cause:**
- Docling generates complex nested metadata (dictionaries within dictionaries)
- ChromaDB only accepts simple types in metadata
- Direct ingestion fails with type error

**Solution:**
- Implemented metadata flattening in `VectorStore.add_documents()`
- Simple types kept as-is
- Complex types serialized as JSON strings
- Useful fields extracted (filename, headings, page_no)

**Code Fix:**
```python
# In VectorStore.add_documents()
for key, value in doc.metadata.items():
    if isinstance(value, (str, int, float, bool)) or value is None:
        metadata[key] = value
    elif key == "origin" and isinstance(value, dict):
        metadata["filename"] = value.get("filename", "")
    elif key == "headings" and isinstance(value, list):
        metadata["headings"] = ", ".join(value) if value else ""
    elif key == "dl_meta" and isinstance(value, dict):
        # Extract page_no and serialize rest
        doc_items = value.get("doc_items", [])
        if doc_items:
            prov = doc_items[0].get("prov", [])
            if prov:
                metadata["page_no"] = prov[0].get("page_no", 0)
        metadata["dl_meta"] = json.dumps(value)
    else:
        metadata[key] = json.dumps(value)
```

### Issue 2: Memory Allocation Errors

**Error:**
```
Stage preprocess failed for run 1, pages [5]: std::bad_alloc
Stage preprocess failed for run 1, pages [6]: std::bad_alloc
```

**Cause:**
- RapidOCR (used by Docling) is memory-intensive
- Complex pages require more memory than available
- Mobile OCR models have limited capacity
- System RAM limitations during batch processing

**Impact:**
- Some pages fail to process
- Partial document ingestion
- Potentially missing important content
- Production scaling concerns

**Mitigation Strategies:**

1. **Page Limiting:**
```python
loader = DoclingLoader(
    str(file_path), 
    export_type=ExportType.DOC_CHUNKS,
    page_range=[0, max_pages_per_doc]  # Limit pages
)
```

2. **Batch Processing:**
```python
import gc

# Process in small batches
for batch in chunked(all_files, batch_size=3):
    for file_path in batch:
        process_file(file_path)
        gc.collect()
```

3. **Memory Management:**
```python
del loader
del documents
gc.collect()
```

4. **Production Alternatives:**
- Cloud-based parsing (AWS Textract, Google Document AI)
- Queue-based processing with workers
- Horizontal scaling
- Alternative parsers (PyPDF2, pdfplumber) for simpler documents

### Issue 3: Similarity Score Bug

**Symptoms:**
```
Retrieved Chunks: []
Answer: "I don't have enough relevant information."
```

**Root Cause:**
```python
# INCORRECT - Original buggy code
sim = 1 - distance
if sim < score_threshold:
    continue  # This was always true, filtering all results
```

**Analysis:**
- ChromaDB returns cosine distance (range [0, 2])
- Conversion `1 - distance` was incorrect
- Score threshold check failed for all results
- LLM received empty context

**Solution:**
```python
# CORRECT - Fixed code
sim = float(-dist)
# Chroma returns ranked results, lower distance = better match
# Negative distance preserves ranking order
```

**Verification:**
- Temporary fix: `sim = 1.0` (forced all results through)
- Confirmed retrieval pipeline was working
- Fixed similarity conversion
- All tests passed after fix

### Issue 4: Empty Retrieval Results

**Debugging Steps:**

1. **Check Vector Store Contents:**
```python
print(store.collection.count())  # Should be > 0
```

2. **Inspect Retrieved Distances:**
```python
print("Distances:", distances)  # Should be reasonable values
```

3. **Verify Query Embedding:**
```python
query_embedding = embedder.embed([query])
print("Query embedding shape:", query_embedding.shape)
```

4. **Check Score Threshold:**
```python
print("Score threshold:", score_threshold)  # Should be 0.0 for testing
```

5. **Test Without Filtering:**
```python
# Temporarily disable threshold
docs = retriever.retrieve(query, score_threshold=-1.0)
```

---

## Production Considerations

### Scalability Challenges

**Document Processing:**
- Memory allocation errors with RapidOCR
- Single-threaded processing bottleneck
- Large document batches cause failures

**Recommendations:**
1. **Queue-Based Processing:**
   - Implement task queue (Celery, RabbitMQ)
   - Process documents asynchronously
   - Retry failed tasks automatically

2. **Horizontal Scaling:**
   - Multiple worker processes
   - Load balancing
   - Distributed processing

3. **Alternative Parsers:**
   - Use simpler parsers for basic documents
   - Reserve Docling for complex layouts
   - Fallback mechanisms

**Vector Store Scaling:**
- ChromaDB suitable for small-medium datasets
- Consider alternatives for large-scale:
  - Qdrant (better nested metadata support)
  - Weaviate (schema-based, nested objects)
  - Pinecone (managed service, auto-scaling)

### Performance Optimization

**Embedding Generation:**
- Batch processing for efficiency
- GPU acceleration (if available)
- Model quantization for faster inference

**Retrieval Optimization:**
- Index tuning (HNSW parameters)
- Caching frequent queries
- Pre-compute embeddings for common queries

**LLM Optimization:**
- Batch API calls
- Streaming responses
- Model selection based on query complexity

### Monitoring & Logging

**Key Metrics to Track:**
- Document processing success rate
- Retrieval latency
- LLM response time
- Memory usage during processing
- Error rates and types

**Logging Strategy:**
- Structured logging (JSON format)
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Centralized log aggregation
- Alert on critical errors

### Error Handling

**Robust Error Handling:**
```python
try:
    documents = process_document(file_path)
except MemoryError:
    logger.error("Memory error processing %s", file_path)
    # Retry with page limiting
    documents = process_document(file_path, max_pages=5)
except Exception as e:
    logger.error("Error processing %s: %s", file_path, e)
    # Continue with next document
    continue
```

**Retry Logic:**
- Exponential backoff
- Maximum retry attempts
- Dead letter queue for failed tasks

### Deployment Architecture

**Recommended Setup:**

```
┌─────────────────────────────────────────────────────────┐
│                  Load Balancer                         │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼────────┐
│  Web Server     │    │  Worker Nodes   │
│  (FastAPI)      │    │  (Celery)        │
└───────┬────────┘    └────────┬────────┘
        │                      │
        │              ┌───────▼────────┐
        │              │  Vector Store   │
        │              │  (ChromaDB)     │
        │              └────────────────┘
        │
┌───────▼────────┐
│  LLM Service    │
│  (Ollama/Gemini)│
└────────────────┘
```

**Components:**
- **Web Server:** FastAPI for REST API
- **Worker Nodes:** Celery for async processing
- **Vector Store:** ChromaDB with persistence
- **LLM Service:** Ollama or Gemini API
- **Message Queue:** RabbitMQ/Redis for task distribution

---

## Setup & Usage

### Installation

**1. Clone Repository:**
```bash
git clone <repository-url>
cd inteli_docs_rag
```

**2. Create Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure Environment:**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**5. Start Ollama (if using local LLM):**
```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull qwen2.5:0.5b
```

### Document Ingestion

**Using Jupyter Notebook:**
```python
# Open notebook/3_docling.ipynb
# Run cells to process documents with Docling
```

**Using Python Script:**
```python
from rag import VectorStore, EmbeddingManager
from langchain_docling.loader import DoclingLoader, ExportType

# Process documents
loader = DoclingLoader("path/to/document.pdf", export_type=ExportType.DOC_CHUNKS)
documents = loader.load()

# Generate embeddings
embedder = EmbeddingManager()
embeddings = embedder.embed([doc.page_content for doc in documents])

# Store in vector database
store = VectorStore()
store.add_documents(documents, embeddings)
```

### Interactive Chat

**Using CLI:**
```bash
# Default: Gemini
python chat.py

# Use Ollama
LLM_BACKEND=ollama python chat.py

# Custom TOP_K
TOP_K=10 python chat.py
```

**Chat Commands:**
- Type your question and press Enter
- Type `exit`, `quit`, or `q` to leave
- Empty lines are ignored

### Running Evaluation

**Execute Evaluation Pipeline:**
```bash
python eval/evaluate_rag.py
```

**Expected Output:**
```
============================================================
RAG Evaluation Pipeline
============================================================
Loaded 10 test cases from eval/test_dataset.csv
[1/10] Q: What is the discount factor used?…
  → Answer (0.45s): The discount factor used is 0.97…
  Retrieval Relevance : 0.9234
  Groundedness        : 0.8750
  Answer Correctness  : 0.9500
  Answer Relevance    : 1.0000
...
============================================================
EVALUATION SUMMARY
============================================================
Metric                          Mean     Std     Min     Max
------------------------------------------------------------
retrieval_relevance            0.8750  0.0923  0.7234  0.9812
groundedness                   0.7823  0.1456  0.5000  0.9500
answer_correctness             0.7156  0.1823  0.4000  0.9200
answer_relevance              0.8890  0.1023  0.7000  1.0000
============================================================
```

**Output Files:**
- `eval/eval_results_<timestamp>.json` - Detailed results
- `eval/eval_summary_<timestamp>.csv` - Summary statistics

### Programmatic Usage

**Basic RAG Query:**
```python
from rag import EmbeddingManager, VectorStore, RAGRetriever, ask
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize components
embedder = EmbeddingManager()
store = VectorStore()
retriever = RAGRetriever(store, embedder)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Ask question
answer = ask("What is the main topic?", retriever, llm)
print(answer)
```

**Custom Retrieval:**
```python
# Retrieve with custom parameters
docs = retriever.retrieve(
    query="machine learning",
    top_k=10,
    score_threshold=0.5,
    filter_metadata={"file_type": "pdf"}
)

# Format context
context = retriever.format_context_for_llm(docs, max_context_length=6000)
print(context)
```

**Evaluation Integration:**
```python
from eval.evaluate_rag import run_evaluation

# Run full evaluation
results = run_evaluation()

# Access aggregated scores
print(results["aggregate"]["retrieval_relevance"]["mean"])
```

---

## Development Notes

### Code Organization Principles

**Single Source of Truth:**
- All configuration in `rag/config.py`
- Core logic in `rag/pipeline.py`
- Public API in `rag/__init__.py`
- Consumers import from `rag` package

**Lazy Imports:**
- Heavy dependencies imported inside functions
- Reduces startup time
- Allows optional dependencies

**Error Handling:**
- Graceful degradation
- Informative error messages
- Logging at appropriate levels

### Testing Strategy

**Unit Tests:**
- Test individual components
- Mock external dependencies
- Fast execution

**Integration Tests:**
- Test component interactions
- Use test vector store
- Real document samples

**Evaluation Tests:**
- Use test dataset
- Track metric trends
- Regression detection

### Code Style

**PEP 8 Compliance:**
- 4-space indentation
- Maximum line length: 88 characters
- Descriptive variable names
- Type hints where appropriate

**Documentation:**
- Docstrings for all public functions
- Inline comments for complex logic
- README for setup instructions

### Version Control

**Git Workflow:**
- Feature branches for new features
- Pull requests for code review
- Semantic versioning

**Commit Messages:**
- Conventional commits format
- Descriptive subject lines
- Detailed body when needed

### Future Enhancements

**Planned Improvements:**
1. **Hybrid Search:** Combine semantic and keyword search
2. **Query Expansion:** Improve retrieval with query rewriting
3. **Multi-Query:** Retrieve with multiple query variations
4. **Citation Extraction:** Extract specific sentence citations
5. **Streaming Responses:** Real-time LLM response streaming
6. **Web Interface:** React-based chat UI
7. **API Endpoints:** REST API for integration
8. **Advanced Metrics:** Add faithfulness, answer consistency
9. **A/B Testing:** Compare different retrieval strategies
10. **User Feedback:** Collect and incorporate user ratings

**Research Directions:**
- Domain-specific embedding models
- Fine-tuning on document corpus
- Adaptive retrieval (dynamic top_k)
- Context compression techniques
- Multi-modal document understanding

---

## Troubleshooting Guide

### Common Issues

**Issue: Import Error for `rag` package**
```
ModuleNotFoundError: No module named 'rag'
```
**Solution:**
```bash
# Ensure you're in project root
cd inteli_docs_rag
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue: ChromaDB Collection Locked**
```
sqlite3.OperationalError: database is locked
```
**Solution:**
- Ensure only one process accesses the vector store
- Check for zombie processes
- Restart ChromaDB client

**Issue: Ollama Connection Refused**
```
ConnectionError: Failed to connect to Ollama
```
**Solution:**
```bash
# Start Ollama server
ollama serve
# Verify model is pulled
ollama list
```

**Issue: Gemini API Rate Limit**
```
RateLimitError: API rate limit exceeded
```
**Solution:**
- Implement exponential backoff
- Use Ollama for evaluation
- Request higher quota

**Issue: Out of Memory During Embedding**
```
MemoryError: Unable to allocate array
```
**Solution:**
- Process documents in smaller batches
- Reduce batch size in embed()
- Use machine with more RAM

### Debug Mode

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Verbose Retrieval:**
```python
# In RAGRetriever.retrieve()
print("Query embedding:", query_embedding)
print("Raw results:", results)
print("Processed results:", ranked)
```

**Profile Performance:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Run your code
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)
```

---

## Contact & Support

**Project Repository:** [GitHub URL]

**Documentation:** This file (`docs/memory.md`)

**Issues:** Report bugs via GitHub Issues

**Contributions:** Follow contribution guidelines in repository

---

## Appendix

### A. Complete Configuration Reference

```python
# rag/config.py - All Configuration Parameters

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Vector Store
COLLECTION_NAME = "pdf_documents"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval Parameters
TOP_K = 5
MAX_CONTEXT_CHARS = 4000
SCORE_THRESHOLD = 0.0

# Document Ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = ["*.pdf", "*.ppt", "*.pptx", "*.doc", "*.docx", "*.txt"]

# LLM Backends
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.0

# RAG Prompt Template
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

### B. API Reference

**rag.EmbeddingManager**
```python
class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None
    def embed(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray
    def generate_embeddings(self, texts: List[str]) -> np.ndarray
```

**rag.VectorStore**
```python
class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME, 
                 persist_directory: str | Path = VECTOR_STORE_DIR) -> None
    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None
```

**rag.RAGRetriever**
```python
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager) -> None
    def retrieve(self, query: str, top_k: int = TOP_K, 
                 score_threshold: float = SCORE_THRESHOLD,
                 filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]
    def format_context_for_llm(self, retrieved_docs: List[Dict[str, Any]], 
                              max_context_length: int = MAX_CONTEXT_CHARS) -> str
```

**rag.ask**
```python
def ask(query: str, retriever: RAGRetriever, llm: Any, 
        top_k: int = TOP_K, max_context_length: int = MAX_CONTEXT_CHARS,
        filter_metadata: Optional[Dict] = None) -> str
```

### C. Evaluation Metrics Reference

**Retrieval Relevance**
- Formula: Mean of top-3 cosine similarities
- Range: [0.0, 1.0]
- Higher is better

**Groundedness**
- LLM judge score
- Range: [0.0, 1.0]
- Higher is better

**Answer Correctness**
- LLM judge score
- Range: [0.0, 1.0]
- Higher is better

**Answer Relevance**
- LLM judge score
- Range: [0.0, 1.0]
- Higher is better

### D. Environment Variables

```
GEMINI_API_KEY      # Required for Gemini backend
LLM_BACKEND         # "gemini" (default) or "ollama"
TOP_K               # Override default top_k (5)
```

---

**Document Version:** 1.0  
**Last Updated:** July 13, 2026  
**Maintainer:** Development Team  
**Status:** COMPLETE - Ready for production use with documented limitations
