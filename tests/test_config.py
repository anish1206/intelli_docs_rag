"""
tests/test_config.py
====================
Unit tests for ``rag/config.py``.

Tests verify that:
1. The three new agentic config variables have correct hard-coded defaults.
2. Environment variable overrides are parsed to the correct Python types.

These tests deliberately import config *inside* each test function so that
``monkeypatch.setenv`` takes effect before the module is re-imported.
``importlib.reload`` is used to force re-evaluation of ``os.getenv`` calls.
"""

import importlib
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_config():
    """Re-import rag.config so that os.getenv picks up monkeypatched values."""
    import rag.config as cfg
    importlib.reload(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------

class TestDefaultValues:
    """Verify hard-coded fallback defaults when env vars are absent."""

    def test_agent_max_retries_default(self, monkeypatch):
        monkeypatch.delenv("AGENT_MAX_RETRIES", raising=False)
        cfg = _reload_config()
        assert cfg.AGENT_MAX_RETRIES == 1

    def test_crag_relevance_threshold_default(self, monkeypatch):
        monkeypatch.delenv("CRAG_RELEVANCE_THRESHOLD", raising=False)
        cfg = _reload_config()
        assert cfg.CRAG_RELEVANCE_THRESHOLD == pytest.approx(0.5)

    def test_history_window_turns_default(self, monkeypatch):
        monkeypatch.delenv("HISTORY_WINDOW_TURNS", raising=False)
        cfg = _reload_config()
        assert cfg.HISTORY_WINDOW_TURNS == 3

    def test_agent_max_retries_is_int(self, monkeypatch):
        monkeypatch.delenv("AGENT_MAX_RETRIES", raising=False)
        cfg = _reload_config()
        assert isinstance(cfg.AGENT_MAX_RETRIES, int)

    def test_crag_relevance_threshold_is_float(self, monkeypatch):
        monkeypatch.delenv("CRAG_RELEVANCE_THRESHOLD", raising=False)
        cfg = _reload_config()
        assert isinstance(cfg.CRAG_RELEVANCE_THRESHOLD, float)

    def test_history_window_turns_is_int(self, monkeypatch):
        monkeypatch.delenv("HISTORY_WINDOW_TURNS", raising=False)
        cfg = _reload_config()
        assert isinstance(cfg.HISTORY_WINDOW_TURNS, int)


# ---------------------------------------------------------------------------
# Environment variable override tests
# ---------------------------------------------------------------------------

class TestEnvironmentOverrides:
    """Verify that string env vars are parsed to the correct Python types."""

    def test_agent_max_retries_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_RETRIES", "3")
        cfg = _reload_config()
        assert cfg.AGENT_MAX_RETRIES == 3
        assert isinstance(cfg.AGENT_MAX_RETRIES, int)

    def test_crag_relevance_threshold_override(self, monkeypatch):
        monkeypatch.setenv("CRAG_RELEVANCE_THRESHOLD", "0.75")
        cfg = _reload_config()
        assert cfg.CRAG_RELEVANCE_THRESHOLD == pytest.approx(0.75)
        assert isinstance(cfg.CRAG_RELEVANCE_THRESHOLD, float)

    def test_history_window_turns_override(self, monkeypatch):
        monkeypatch.setenv("HISTORY_WINDOW_TURNS", "5")
        cfg = _reload_config()
        assert cfg.HISTORY_WINDOW_TURNS == 5
        assert isinstance(cfg.HISTORY_WINDOW_TURNS, int)

    def test_agent_max_retries_zero(self, monkeypatch):
        """Edge case: retries disabled."""
        monkeypatch.setenv("AGENT_MAX_RETRIES", "0")
        cfg = _reload_config()
        assert cfg.AGENT_MAX_RETRIES == 0

    def test_crag_relevance_threshold_extremes(self, monkeypatch):
        """Boundary values: exactly 0.0 and 1.0 should parse cleanly."""
        for value in ("0.0", "1.0"):
            monkeypatch.setenv("CRAG_RELEVANCE_THRESHOLD", value)
            cfg = _reload_config()
            assert isinstance(cfg.CRAG_RELEVANCE_THRESHOLD, float)
