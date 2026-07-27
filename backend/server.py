import logging
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from rag.pipeline import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
)

from rag.config import (
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
)

from backend.chat_memory import ChatMemory


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Personal RAG API",
    description="RAG API for personal notes and documents",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    question: str


# =========================================================
# INITIALIZE RAG COMPONENTS
# =========================================================

logger.info("Loading embedding model...")

embedder = EmbeddingManager()

logger.info("Loading vector store...")

store = VectorStore(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_STORE_DIR,
)

logger.info("Creating retriever...")

retriever = RAGRetriever(
    vector_store=store,
    embedding_manager=embedder,
)


# =========================================================
# INITIALIZE LLM
# =========================================================

from langchain_ollama import ChatOllama

from rag.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)


# =========================================================
# INITIALIZE MEMORY
# =========================================================

memory = ChatMemory()


# =========================================================
# HEALTH CHECK
# =========================================================

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

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    # -----------------------------------------------------
    # 1. Clean and validate question
    # -----------------------------------------------------

    question = request.question.strip()

    if not question:

        return {
            "error": "Question cannot be empty."
        }

    logger.info(
        "Received question: %s",
        question
    )

    # -----------------------------------------------------
    # 2. Check Redis cache
    # -----------------------------------------------------

    cached_response = memory.get_cached_answer(
        question
    )

    if cached_response is not None:

        logger.info(
            "Cache hit. Returning cached response."
        )

        return {
            **cached_response,
            "from_cache": True,
        }

    logger.info(
        "Cache miss. Running RAG pipeline."
    )

    # -----------------------------------------------------
    # 3. Retrieve relevant documents
    # -----------------------------------------------------

    docs = retriever.retrieve(
        question
    )

    # -----------------------------------------------------
    # 4. Format context
    # -----------------------------------------------------

    context = retriever.format_context(
        docs
    )

    # -----------------------------------------------------
    # 5. Create prompt
    # -----------------------------------------------------

    prompt = f"""
You are a helpful assistant that answers questions based on the provided context.

The context contains information from various documents with source citations.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question using only the provided context.
- If the answer is not in the context, say you don't have enough information.
- Include source citations in your answer when relevant.
- Be specific and accurate.
- If multiple documents provide information, synthesize them coherently.
"""

    # -----------------------------------------------------
    # 6. Generate answer using LLM
    # -----------------------------------------------------

    response = llm.invoke(
        prompt
    )

    answer = response.content

    # -----------------------------------------------------
    # 7. Create complete response object
    # -----------------------------------------------------

    result = {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": doc["similarity"],
            }
            for doc in docs
        ],
    }

    # -----------------------------------------------------
    # 8. Store conversation history
    # -----------------------------------------------------

    memory.add_message(
        role="user",
        content=question,
    )

    memory.add_message(
        role="assistant",
        content=answer,
    )

    # -----------------------------------------------------
    # 9. Store complete response in Redis cache
    # -----------------------------------------------------

    memory.cache_answer(
        question=question,
        response=result,
    )

    logger.info(
        "Answer generated and cached."
    )

    # -----------------------------------------------------
    # 10. Return response to frontend
    # -----------------------------------------------------

    return {
        **result,
        "from_cache": False,

    }