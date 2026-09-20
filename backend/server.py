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
    session_id: str = "default"


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
# INITIALIZE MEMORY & AGENT GRAPH
# =========================================================

memory = ChatMemory()

from rag.agent.graph import build_agentic_rag_graph
logger.info("Building Agentic RAG graph...")
agent_app = build_agentic_rag_graph(retriever=retriever, llm=llm)



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
# SESSION ENDPOINTS
# =========================================================

@app.get("/sessions")
def get_sessions():
    return memory.get_all_sessions()

@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    history = memory.get_history(session_id)
    return {"history": history}

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
    # 1.5 Handle session registration
    # -----------------------------------------------------

    session_id = request.session_id
    # If this is a new session, history will be empty
    if not memory.get_history(session_id):
        # Generate a short title from the question (first 5 words)
        words = question.split()
        title = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
        memory.register_session(session_id, title)
    else:
        # Just update the timestamp
        title = memory.redis.get(f"chat:{session_id}:title") or "New Chat"
        memory.register_session(session_id, title)

    # -----------------------------------------------------
    # 1.6 Handle generic greetings
    # -----------------------------------------------------
    
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good evening", "howdy", "sup"}
    if question.lower() in greetings:
        answer = "Hello! I am your Intelli Docs assistant. How can I help you with your documents today?"
        
        # Store conversation history
        memory.add_message(role="user", content=question, session_id=session_id)
        memory.add_message(role="assistant", content=answer, session_id=session_id)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "from_cache": False,
        }

    # -----------------------------------------------------
    # 2. Check Redis cache
    # -----------------------------------------------------

    cached_response = memory.get_cached_answer(
        question, session_id
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
    # 3. Invoke Agent Graph
    # -----------------------------------------------------

    recent_history = memory.get_recent_history_formatted(session_id=session_id)

    initial_state = {
        "question": question,
        "session_id": session_id,
        "history": recent_history,
        "standalone_query": question,
        "tool_choice": None,
        "tool_output": None,
        "documents": [],
        "retrieval_retry_count": 0,
        "hallucination_retry_count": 0,
        "is_relevant": False,
        "is_grounded": False,
        "final_answer": "",
        "sources": [],
    }

    final_state = agent_app.invoke(initial_state)

    answer = final_state.get("final_answer", "")
    sources = final_state.get("sources", [])

    agent_meta = {
        "standalone_query": final_state.get("standalone_query", question),
        "tool_choice": final_state.get("tool_choice"),
        "retrieval_retried": final_state.get("retrieval_retry_count", 0) > 0,
        "guard_passed": final_state.get("is_grounded", True),
    }

    # -----------------------------------------------------
    # 4. Create complete response object
    # -----------------------------------------------------

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "agent_metadata": agent_meta
    }

    # -----------------------------------------------------
    # 5. Store conversation history
    # -----------------------------------------------------

    memory.add_message(
        role="user",
        content=question,
        session_id=session_id,
    )

    memory.add_message(
        role="assistant",
        content=answer,
        session_id=session_id,
    )

    # -----------------------------------------------------
    # 6. Store complete response in Redis cache
    # -----------------------------------------------------

    memory.cache_answer(
        question=question,
        response=result,
        session_id=session_id,
    )

    logger.info(
        "Answer generated and cached."
    )

    # -----------------------------------------------------
    # 7. Return response to frontend
    # -----------------------------------------------------

    return {
        **result,
        "from_cache": False,
    }