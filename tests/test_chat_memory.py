"""
tests/test_chat_memory.py
=========================
Unit tests for backend/chat_memory.py.

Uses the fakeredis fixtures defined in conftest.py so that tests run instantly
without needing a live Redis server running.
"""

import pytest
from backend.chat_memory import ChatMemory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chat_memory(fake_redis_client):
    """Returns a ChatMemory instance pointing at the flushed fakeredis client."""
    # We construct ChatMemory and immediately swap its internal redis client
    # with the flushed fake one.
    mem = ChatMemory(redis_host="dummy")
    mem.redis = fake_redis_client
    return mem


# ---------------------------------------------------------------------------
# get_recent_history_formatted tests
# ---------------------------------------------------------------------------

class TestGetRecentHistoryFormatted:

    def test_empty_history(self, chat_memory):
        """Should return an empty string when the session has no history."""
        result = chat_memory.get_recent_history_formatted("session-1", max_turns=2)
        assert result == ""

    def test_fewer_messages_than_max_turns(self, chat_memory):
        """Should return all available messages formatted correctly."""
        chat_memory.add_message("user", "Hello", "session-1")
        chat_memory.add_message("assistant", "Hi there", "session-1")

        # Requesting 2 turns (4 messages), but only 1 turn (2 messages) exists
        result = chat_memory.get_recent_history_formatted("session-1", max_turns=2)
        
        expected = "User: Hello\nAssistant: Hi there"
        assert result == expected

    def test_exact_max_turns(self, chat_memory):
        """Should return exactly the requested number of turns."""
        chat_memory.add_message("user", "Q1", "session-1")
        chat_memory.add_message("assistant", "A1", "session-1")
        chat_memory.add_message("user", "Q2", "session-1")
        chat_memory.add_message("assistant", "A2", "session-1")

        result = chat_memory.get_recent_history_formatted("session-1", max_turns=2)
        
        expected = "User: Q1\nAssistant: A1\nUser: Q2\nAssistant: A2"
        assert result == expected

    def test_more_messages_than_max_turns(self, chat_memory):
        """Should slice off the oldest turns and keep only the most recent max_turns."""
        # Turn 1
        chat_memory.add_message("user", "Old Q1", "session-1")
        chat_memory.add_message("assistant", "Old A1", "session-1")
        # Turn 2
        chat_memory.add_message("user", "Recent Q2", "session-1")
        chat_memory.add_message("assistant", "Recent A2", "session-1")
        # Turn 3
        chat_memory.add_message("user", "Recent Q3", "session-1")
        chat_memory.add_message("assistant", "Recent A3", "session-1")

        # Requesting 2 turns should drop Turn 1 completely
        result = chat_memory.get_recent_history_formatted("session-1", max_turns=2)
        
        expected = "User: Recent Q2\nAssistant: Recent A2\nUser: Recent Q3\nAssistant: Recent A3"
        assert result == expected

    def test_fallback_to_config_default(self, chat_memory, monkeypatch):
        """Should use HISTORY_WINDOW_TURNS from rag.config if max_turns is None."""
        monkeypatch.setattr("rag.config.HISTORY_WINDOW_TURNS", 1)
        
        # Turn 1
        chat_memory.add_message("user", "Old Q1", "session-1")
        chat_memory.add_message("assistant", "Old A1", "session-1")
        # Turn 2
        chat_memory.add_message("user", "Recent Q2", "session-1")
        chat_memory.add_message("assistant", "Recent A2", "session-1")

        result = chat_memory.get_recent_history_formatted("session-1", max_turns=None)
        
        expected = "User: Recent Q2\nAssistant: Recent A2"
        assert result == expected

    def test_odd_number_of_messages(self, chat_memory):
        """Should handle cases where the conversation ends on a user message without assistant reply."""
        chat_memory.add_message("user", "Q1", "session-1")
        chat_memory.add_message("assistant", "A1", "session-1")
        chat_memory.add_message("user", "Q2 (interrupted)", "session-1")

        # Requesting 2 turns, we have 3 messages
        result = chat_memory.get_recent_history_formatted("session-1", max_turns=2)
        
        expected = "User: Q1\nAssistant: A1\nUser: Q2 (interrupted)"
        assert result == expected

    def test_empty_content_filtered(self, chat_memory):
        """Should silently skip over messages with empty content."""
        chat_memory.add_message("user", "Hello", "session-1")
        chat_memory.add_message("assistant", "", "session-1") # empty
        chat_memory.add_message("user", "Are you there?", "session-1")
        
        result = chat_memory.get_recent_history_formatted("session-1", max_turns=2)
        
        expected = "User: Hello\nUser: Are you there?"
        assert result == expected
