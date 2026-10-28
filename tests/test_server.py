import pytest
from fastapi.testclient import TestClient
from backend.llm_client import LlmRequestError
from unittest.mock import MagicMock, patch
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    def test_llm_client_has_finite_timeouts(self):
        from backend import server
        assert server.llm.connect_timeout > 0
        assert server.llm.read_timeout > 0
        assert server.llm.request_timeout > 0

    @pytest.mark.parametrize("timed_out,status", [(True, 504), (False, 502)])
    @patch("backend.server.agent_app")
    def test_llm_failure_is_an_api_error(self, graph, timed_out, status):
        graph.invoke.side_effect = LlmRequestError(timed_out=timed_out)
        response = client.post("/chat", json={"question": "Explain this", "session_id": "failure"})
        assert response.status_code == status
        assert memory.get_history("failure") == []

    def test_real_graph_timeout_returns_504(self, monkeypatch):
        from backend import server
        from backend.llm_client import BoundedOllama
        from rag.agent.graph import build_agentic_rag_graph

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                time.sleep(0.3)
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{}\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        provider = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setenv("OLLAMA_MAX_RETRIES", "0")
            monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "0.05")
            llm = BoundedOllama("model", f"http://127.0.0.1:{provider.server_port}", 0)
            monkeypatch.setattr(server, "agent_app", build_agentic_rag_graph(MagicMock(), llm))
            response = client.post("/chat", json={"question": "Explain this", "session_id": "timeout"})
            assert response.status_code == 504
            assert memory.get_history("timeout") == []
        finally:
            provider.shutdown()
            provider.server_close()
            thread.join()

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
