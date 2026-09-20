"""
tests/test_e2e_agent.py
========================
Offline end-to-end integration tests for the Agentic RAG workflow.
Uses pytest and mocked LLM/retriever dependencies to test all 4 core agentic behaviors:
1. Multi-turn context rewriting
2. Calculator tool execution
3. List notes tool execution
4. CRAG & Hallucination Guard retry cap enforcement
"""

import pytest
from unittest.mock import MagicMock
from rag.agent.graph import build_agentic_rag_graph

class TestE2EAgentWorkflows:

    def test_multi_turn_context_rewriting(self):
        """Verify multi-turn conversation history triggers rewriter to produce a standalone query."""
        retriever = MagicMock()
        retriever.format_context_for_llm.return_value = "Quicksort worst-case is O(n^2)."

        llm = MagicMock()
        # Mock LLM responses in sequence:
        # 1. Rewriter node
        # 2. Router node -> search
        # 3. Grade node -> relevant
        # 4. Generator node -> final answer
        # 5. Guard node -> grounded
        llm.invoke.side_effect = [
            MagicMock(content="What is the worst case complexity of quicksort?"),
            MagicMock(content='{"tool": "search"}'),
            MagicMock(content='{"relevant": true}'),
            MagicMock(content="The worst-case time complexity of quicksort is O(n^2)."),
            MagicMock(content='{"grounded": true}')
        ]

        app = build_agentic_rag_graph(retriever, llm)

        initial_state = {
            "question": "What is its worst case complexity?",
            "history": "User: What is quicksort?\nAssistant: Quicksort is a divide and conquer sorting algorithm.",
        }

        final_state = app.invoke(initial_state)

        assert final_state["standalone_query"] == "What is the worst case complexity of quicksort?"
        assert final_state["tool_choice"] == "search"
        assert final_state["is_relevant"] is True
        assert final_state["is_grounded"] is True
        assert "O(n^2)" in final_state["final_answer"]

    def test_tool_execution_calculator(self):
        """Verify math expressions route to calculator tool and produce accurate results."""
        retriever = MagicMock()
        llm = MagicMock()

        # LLM sequence:
        # 1. Rewriter node (skipped/passthrough since history is empty or returns query)
        # 2. Router node -> calculate
        # 3. Generator node -> Final Answer using tool output
        # 4. Guard node -> grounded
        llm.invoke.side_effect = [
            MagicMock(content='{"tool": "calculate"}'),
            MagicMock(content="The result of 15 * 8 + 3 is 123."),
            MagicMock(content='{"grounded": true}')
        ]

        app = build_agentic_rag_graph(retriever, llm)

        initial_state = {
            "question": "15 * 8 + 3",
            "history": "",
        }

        final_state = app.invoke(initial_state)

        assert final_state["tool_choice"] == "calculate"
        assert final_state["tool_output"] == "123"
        assert "123" in final_state["final_answer"]

    def test_tool_execution_list_notes(self):
        """Verify document listing queries route to list_docs tool and return filenames."""
        retriever = MagicMock()
        # Mock retriever vector_store.get behavior
        retriever.vector_store.get.return_value = {
            "metadatas": [{"source": "lecture1.pdf"}, {"source": "syllabus.md"}]
        }

        llm = MagicMock()
        # LLM sequence:
        # 1. Router node -> list_docs
        # 2. Generator node -> Final Answer summarizing available notes
        # 3. Guard node -> grounded
        llm.invoke.side_effect = [
            MagicMock(content='{"tool": "list_docs"}'),
            MagicMock(content="Available documents: lecture1.pdf, syllabus.md"),
            MagicMock(content='{"grounded": true}')
        ]

        app = build_agentic_rag_graph(retriever, llm)

        initial_state = {
            "question": "What documents do you have available?",
            "history": "",
        }

        final_state = app.invoke(initial_state)

        assert final_state["tool_choice"] == "list_docs"
        assert final_state["tool_output"] == ["lecture1.pdf", "syllabus.md"]
        assert "lecture1.pdf" in final_state["final_answer"]

    def test_crag_retry_cap(self, monkeypatch):
        """Verify CRAG retrieval loops respect AGENT_MAX_RETRIES without infinite loops."""
        monkeypatch.setattr("rag.agent.nodes.AGENT_MAX_RETRIES", 2)

        retriever = MagicMock()
        llm = MagicMock()

        # LLM sequence:
        # 1. Router -> search
        # 2. Grade 1 -> relevant: False
        # 3. Reformulate 1 -> query_retry_1 (retrieval_retry_count becomes 1)
        # 4. Grade 2 -> relevant: False
        # 5. Reformulate 2 -> query_retry_2 (retrieval_retry_count becomes 2)
        # 6. Grade 3 -> relevant: False
        # (retrieval_retry_count = 2 >= AGENT_MAX_RETRIES 2, crag_decision proceeds to generate)
        # 7. Generator -> draft answer
        # 8. Guard -> grounded
        llm.invoke.side_effect = [
            MagicMock(content='{"tool": "search"}'),
            MagicMock(content='{"relevant": false}'),
            MagicMock(content="query_retry_1"),
            MagicMock(content='{"relevant": false}'),
            MagicMock(content="query_retry_2"),
            MagicMock(content='{"relevant": false}'),
            MagicMock(content="Fallback answer after max retries."),
            MagicMock(content='{"grounded": true}')
        ]

        app = build_agentic_rag_graph(retriever, llm)

        initial_state = {
            "question": "Obscure topic query",
            "history": "",
        }

        final_state = app.invoke(initial_state)

        assert final_state["retrieval_retry_count"] == 2
        assert final_state["is_relevant"] is False
        assert final_state["final_answer"] == "Fallback answer after max retries."

    def test_hallucination_guard_retry_cap(self, monkeypatch):
        """Verify Hallucination Guard loops respect AGENT_MAX_RETRIES without infinite loops."""
        monkeypatch.setattr("rag.agent.nodes.AGENT_MAX_RETRIES", 2)

        retriever = MagicMock()
        llm = MagicMock()

        # LLM sequence:
        # 1. Router -> search
        # 2. Grade -> relevant: True
        # 3. Generator 1 -> Draft 1
        # 4. Guard 1 -> grounded: False (hallucination_retry_count becomes 1)
        # 5. Generator 2 -> Draft 2
        # 6. Guard 2 -> grounded: False (hallucination_retry_count becomes 2)
        # (hallucination_retry_count = 2 >= AGENT_MAX_RETRIES 2, guard_decision returns pass)
        llm.invoke.side_effect = [
            MagicMock(content='{"tool": "search"}'),
            MagicMock(content='{"relevant": true}'),
            MagicMock(content="Draft answer 1"),
            MagicMock(content='{"grounded": false}'),
            MagicMock(content="Draft answer 2"),
            MagicMock(content='{"grounded": false}'),
        ]

        app = build_agentic_rag_graph(retriever, llm)

        initial_state = {
            "question": "Sample topic",
            "history": "",
        }

        final_state = app.invoke(initial_state)

        assert final_state["hallucination_retry_count"] == 2
        assert final_state["is_grounded"] is False
        assert final_state["final_answer"] == "Draft answer 2"

