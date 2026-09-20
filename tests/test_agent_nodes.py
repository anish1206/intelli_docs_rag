import pytest
from unittest.mock import MagicMock
import os
from rag.agent.nodes import (
    create_rewriter_node,
    route_decision,
    crag_decision,
    guard_decision,
    clean_json_response
)

class TestAgentNodes:

    def test_clean_json_response(self):
        # Clean JSON
        assert clean_json_response('{"test": true}') == {"test": True}
        
        # Markdown wrapped
        assert clean_json_response('```json\n{"test": true}\n```') == {"test": True}
        assert clean_json_response('```\n{"test": false}\n```') == {"test": False}
        
        # Corrupted strings
        assert clean_json_response('not json') == {}

    def test_rewrite_query_node_empty_history(self):
        llm = MagicMock()
        node = create_rewriter_node(llm)
        
        state = {"question": "What is AI?", "history": ""}
        result = node(state)
        
        # LLM should not be called
        llm.invoke.assert_not_called()
        assert result["standalone_query"] == "What is AI?"

    def test_crag_decision(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_RETRIES", "1")
        
        # Is relevant -> proceed
        assert crag_decision({"is_relevant": True, "retrieval_retry_count": 0}) == "proceed"
        
        # Not relevant, retry count 0 -> retry
        assert crag_decision({"is_relevant": False, "retrieval_retry_count": 0}) == "retry"
        
        # Not relevant, retry count 1 (max) -> proceed
        assert crag_decision({"is_relevant": False, "retrieval_retry_count": 1}) == "proceed"

    def test_guard_decision(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_RETRIES", "1")
        
        # Is grounded -> pass
        assert guard_decision({"is_grounded": True, "hallucination_retry_count": 0}) == "pass"
        
        # Not grounded, retry count 0 -> regenerate
        assert guard_decision({"is_grounded": False, "hallucination_retry_count": 0}) == "regenerate"
        
        # Not grounded, retry count 1 (max) -> pass
        assert guard_decision({"is_grounded": False, "hallucination_retry_count": 1}) == "pass"
