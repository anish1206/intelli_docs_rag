import pytest
from unittest.mock import MagicMock
from rag.agent.tools import calculate, search_notes, list_available_notes

class TestAgentTools:
    
    def test_calculate_valid(self):
        assert calculate("2 + 2") == "4"
        assert calculate("(85 * 0.2) + (90 * 0.8)") == "89.0"
        
    def test_calculate_error_handling(self):
        # Division by zero
        assert "Division by zero" in calculate("10 / 0")
        
        # Syntax error
        assert "syntax" in calculate("10 / * 2").lower()
        
        # Other exception (e.g., trying to use forbidden functions or unknown vars)
        assert "Error evaluating expression" in calculate("__import__('os').system('ls')")
        
    def test_search_notes(self):
        retriever = MagicMock()
        # Mock the retrieve method
        retriever.retrieve.return_value = [{"page_content": "dummy text", "metadata": {"source": "doc1.pdf"}}]
        
        results = search_notes(retriever, "test query")
        
        assert len(results) == 1
        assert results[0]["page_content"] == "dummy text"
        retriever.retrieve.assert_called_once_with(query="test query", top_k=5)
        
    def test_list_available_notes(self):
        retriever = MagicMock()
        vector_store = MagicMock()
        retriever.vector_store = vector_store
        
        vector_store.get.return_value = {
            "metadatas": [
                {"source": "doc1.pdf"},
                {"source": "doc2.pdf"},
                {"source": "doc1.pdf"} # duplicate should be removed
            ]
        }
        
        docs = list_available_notes(retriever)
        assert docs == ["doc1.pdf", "doc2.pdf"]
