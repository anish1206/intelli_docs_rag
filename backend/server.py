"""
backend/server.py

Web API for the Intelli Docs RAG application.
"""

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# RAG imports
# --------------------------------------------------

from rag.pipeline import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    load_env,
)

from rag.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_env()


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Intelli Docs RAG API",
    description="Backend API for querying personal documents using RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str


# --------------------------------------------------
# Initialize RAG components
# --------------------------------------------------

print("Loading embedding model...")

embedder = EmbeddingManager()

print("Connecting to vector store...")

store = VectorStore()

retriever = RAGRetriever(
    vector_store=store,
    embedding_manager=embedder,
)


# --------------------------------------------------
# Initialize LLM
# --------------------------------------------------

print("Connecting to Ollama...")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)

print("RAG backend initialized successfully.")


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Intelli Docs RAG API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store_documents": store.collection.count(),
    }


@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        return {
            "error": "Question cannot be empty."
        }

    # 1. Retrieve relevant documents
    docs = retriever.retrieve(question)

    # 2. Format retrieved documents into context
    context = retriever.format_context(docs)

    # 3. Create prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
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

"""

    # 4. Generate answer
    response = llm.invoke(prompt)

    # 5. Return response to frontend
    return {
        "question": question,
        "answer": response.content,
        "sources": [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": doc["similarity"],
            }
            for doc in docs
        ],
    }