import pytest
from unittest.mock import MagicMock
from rag.agent.graph import build_agentic_rag_graph

class TestAgentGraph:

    def test_build_agentic_rag_graph(self):
        retriever = MagicMock()
        llm = MagicMock()
        
        # Build the graph
        app = build_agentic_rag_graph(retriever, llm)
        
        assert app is not None
        # It's a CompiledStateGraph
        assert hasattr(app, "invoke")
        assert hasattr(app, "ainvoke")
        
    def test_direct_execution_path_calculator(self):
        retriever = MagicMock()
        llm = MagicMock()
        
        # Mock LLM to simulate the router choosing 'calculate'
        llm.invoke.side_effect = [
            # 1. Router
            MagicMock(content='{"tool": "calculate"}'),
            # 2. Generate Answer (after calculate tool output is injected)
            MagicMock(content="The answer is 4"),
            # 3. Guard
            MagicMock(content='{"grounded": true}')
        ]
        
        app = build_agentic_rag_graph(retriever, llm)
        
        initial_state = {
            "question": "2 + 2",
            "history": "",
        }
        
        final_state = app.invoke(initial_state)
        
        assert final_state["tool_choice"] == "calculate"
        assert final_state["tool_output"] == "4"
        assert "The answer is 4" in final_state["final_answer"]

    def test_standard_search_path(self, monkeypatch):
        # Override retry limits to ensure we don't loop infinitely in tests
        monkeypatch.setenv("AGENT_MAX_RETRIES", "1")
        
        retriever = MagicMock()
        # Mock retriever format_context_for_llm or it will just join
        retriever.format_context_for_llm.return_value = "Document context here"
        
        llm = MagicMock()
        llm.invoke.side_effect = [
            # 1. Router
            MagicMock(content='{"tool": "search"}'),
            # 2. Grade
            MagicMock(content='{"relevant": true}'),
            # 3. Generate
            MagicMock(content="Draft answer based on docs"),
            # 4. Guard
            MagicMock(content='{"grounded": true}')
        ]
        
        app = build_agentic_rag_graph(retriever, llm)
        
        initial_state = {
            "question": "What is AI?",
            "history": "",
        }
        
        final_state = app.invoke(initial_state)
        
        assert final_state["tool_choice"] == "search"
        assert final_state["is_relevant"] is True
        assert final_state["is_grounded"] is True
        assert final_state["final_answer"] == "Draft answer based on docs"
        
    def test_crag_retry_path(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_RETRIES", "1")
        
        retriever = MagicMock()
        
        llm = MagicMock()
        llm.invoke.side_effect = [
            # 1. Router -> search
            MagicMock(content='{"tool": "search"}'),
            # 2. Grade -> not relevant (triggers reformulate)
            MagicMock(content='{"relevant": false}'),
            # 3. Reformulate -> "AI definition"
            MagicMock(content="AI definition"),
            # 4. Grade -> now relevant
            MagicMock(content='{"relevant": true}'),
            # 5. Generate -> draft
            MagicMock(content="Draft answer"),
            # 6. Guard -> grounded
            MagicMock(content='{"grounded": true}')
        ]
        
        app = build_agentic_rag_graph(retriever, llm)
        
        initial_state = {
            "question": "What is AI?",
            "history": "",
        }
        
        final_state = app.invoke(initial_state)
        
        assert final_state["tool_choice"] == "search"
        assert final_state["retrieval_retry_count"] == 1
        assert final_state["is_relevant"] is True
        assert final_state["standalone_query"] == "AI definition"
