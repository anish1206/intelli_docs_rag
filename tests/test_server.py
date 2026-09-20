import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys

# Mock out heavy components before importing server
sys.modules['langchain_ollama'] = MagicMock()
sys.modules['rag.pipeline'] = MagicMock()

# Import the memory instance first, and override its redis connection before tests run
from backend.server import memory, app
client = TestClient(app)

@pytest.fixture(autouse=True)
def override_memory_redis(fake_redis_client):
    memory.redis = fake_redis_client
    yield
    memory.redis = None

class TestServer:

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_chat_greeting_fast_path(self):
        # A greeting should bypass LLM completely
        response = client.post("/chat", json={"question": "Hello", "session_id": "test-session"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["from_cache"] is False
        assert "Hello!" in data["answer"]
        
        # Verify it was added to memory
        history = memory.get_history("test-session")
        assert len(history) == 2
        assert history[0]["content"] == "Hello"

    @patch("backend.server.agent_app")
    def test_chat_agent_graph_execution(self, mock_agent_app):
        # Mock the agent graph execution
        mock_agent_app.invoke.return_value = {
            "final_answer": "This is an agentic answer.",
            "sources": [],
            "standalone_query": "What is agentic RAG?",
            "tool_choice": "search",
            "retrieval_retry_count": 1,
            "is_grounded": True
        }
        
        response = client.post("/chat", json={
            "question": "What is agentic RAG?", 
            "session_id": "agent-session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is an agentic answer."
        assert data["from_cache"] is False
        assert "agent_metadata" in data
        
        meta = data["agent_metadata"]
        assert meta["tool_choice"] == "search"
        assert meta["retrieval_retried"] is True
        assert meta["guard_passed"] is True
        assert meta["standalone_query"] == "What is agentic RAG?"
        
    @patch("backend.server.agent_app")
    def test_chat_redis_cache_hit(self, mock_agent_app):
        # Setup: seed the cache
        memory.cache_answer(
            question="What is cache?",
            response={
                "question": "What is cache?",
                "answer": "Cached answer.",
                "sources": [],
                "agent_metadata": {}
            },
            session_id="cache-session"
        )
        
        response = client.post("/chat", json={
            "question": "What is cache?", 
            "session_id": "cache-session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # It should come from cache
        assert data["from_cache"] is True
        assert data["answer"] == "Cached answer."
        
        # Agent graph should not be invoked
        mock_agent_app.invoke.assert_not_called()
