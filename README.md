<div align="center">

  <img width="501" height="112" alt="IntelliDocs Logo" src="https://github.com/user-attachments/assets/245a3c4d-4707-48e3-931d-396c109c7555" />

A local RAG (Retrieval-Augmented Generation) pipeline for querying your documents.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=databricks&logoColor=FF3621)
![Docling](https://img.shields.io/badge/Docling-000000?style=for-the-badge&logo=ibm&logoColor=FFB800)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

</div>

---

## Features

- **Multi-Format Document Parsing**: Support for PDF, PPT, Word, and Text files using Docling
- **Semantic Search**: Dense embeddings with SentenceTransformer for accurate retrieval
- **Persistent Vector Storage**: ChromaDB for scalable and persistent document storage
- **Multiple LLM Backends**: Support for Google Gemini and local Ollama models
- **Custom Evaluation Pipeline**: Four metrics (Retrieval Relevance, Groundedness, Answer Correctness, Answer Relevance)
- **Source Citations**: Automatic source attribution in generated responses
- **Web UI**: Modern React-based interface for interactive document querying
- **REST API**: FastAPI backend for seamless integration and scalability

## Project Structure

```text
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
│   └── memory.md                # Complete project documentation
├── notebook/                     # Jupyter notebooks
│   ├── 1_document.ipynb         # Initial document processing
│   ├── 2_.ipynb                 # Additional experiments
│   └── 3_docling.ipynb          # Docling integration
├── backend/                      # FastAPI backend server
│   └── server.py                # REST API endpoints
├── frontend/                     # React web interface
│   ├── src/                     # React source code
│   │   ├── components/          # UI components
│   │   │   ├── ChatMessage.jsx
│   │   │   ├── Header.jsx
│   │   │   ├── QuestionInput.jsx
│   │   │   ├── SourceCard.jsx
│   │   │   └── StatusBadge.jsx
│   │   ├── services/            # API service layer
│   │   │   └── api.js
│   │   ├── App.jsx              # Main application
│   │   └── main.jsx             # Entry point
│   ├── package.json             # Frontend dependencies
│   └── vite.config.js           # Vite configuration
├── chat.py                       # Interactive CLI interface
├── rebuild_vector_store.py      # Vector store indexer
├── requirements.txt              # Python dependencies
├── package.json                  # Root dependencies
├── .env.example                 # Environment variables template
└── .env                         # Actual environment variables (gitignored)
```

## How to Use

### Prerequisites

- Python 3.8 or higher
- Node.js 18+ and npm (for frontend)
- Git (for cloning the repository)
- Ollama (optional, for local LLM)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd inteli_docs_rag
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This includes FastAPI, Uvicorn, and all RAG pipeline dependencies.

4. **Install frontend dependencies**

   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY if using Gemini
   ```

6. **Start Ollama (if using local LLM)**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull qwen2.5:0.5b
   ```
---

### Web UI

For a modern web interface, use the React-based frontend:

**1. Start the backend server:**

```bash
# Terminal 1
uvicorn backend.server:app --reload
```

The backend will be available at `http://localhost:8000`

**2. Start the frontend development server:**

```bash
# Terminal 2
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

**3. Access the web interface:**

Open your browser and navigate to `http://localhost:5173`

**Web UI Features:**

- Real-time health status monitoring
- Interactive chat interface with message history
- Source citation display with similarity scores
- Responsive design with modern UI
- Error handling and retry mechanisms
- Backend connection status indicator

**API Endpoints:**

- `GET /` - API root endpoint
- `GET /health` - Health check with document count
- `POST /chat` - Submit query to RAG pipeline

**Build for production:**

```bash
cd frontend
npm run build
```

The optimized build will be in `frontend/dist/`

**Environment Variables for Frontend:**

Create a `.env` file in the `frontend/` directory (optional):

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

If not set, the frontend defaults to `http://localhost:8000`

## Quick Start


```bash
# Terminal 1 - Start backend
uvicorn backend.server:app --reload

# Terminal 2 - Start frontend
cd frontend
npm run dev

# Terminal 3 - Start Ollama (if using local LLM)
ollama serve
ollama pull qwen2.5:0.5b
```

Then open `http://localhost:5173` in your browser.

---
### Running Evaluation

Evaluate the RAG pipeline performance:

```bash
python eval/evaluate_rag.py
```

This will:

- Load test questions from `eval/test_dataset.csv`
- Generate answers using the RAG pipeline
- Compute four evaluation metrics:
  - Retrieval Relevance
  - Groundedness
  - Answer Correctness
  - Answer Relevance
- Save results to `eval/eval_results_<timestamp>.json`
- Generate summary in `eval/eval_summary_<timestamp>.csv`

---
### Configuration

Key configuration parameters are in `rag/config.py`:

- `TOP_K`: Number of chunks to retrieve (default: 5)
- `MAX_CONTEXT_CHARS`: Maximum context length for LLM (default: 4000)
- `EMBEDDING_MODEL`: SentenceTransformer model (default: all-MiniLM-L6-v2)
- `GEMINI_MODEL`: Gemini model name (default: gemini-2.5-flash)
- `OLLAMA_MODEL`: Ollama model name (default: qwen2.5:0.5b)

Override via environment variables:

```bash
TOP_K=10 python chat.py
LLM_BACKEND=ollama python chat.py
```
