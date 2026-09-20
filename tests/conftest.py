"""
tests/conftest.py
=================
Shared pytest fixtures for the Intelli Docs RAG test suite.

All fixtures are session-scoped where possible to avoid repeated setup
overhead, and use fakeredis so no live Redis server is required.
"""

import json
import pytest
import fakeredis


# ---------------------------------------------------------------------------
# fakeredis fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_redis_server():
    """A single in-process fakeredis server shared across the entire session.

    Using a session-scoped server (rather than creating a new one per test)
    means the underlying data store is shared, so individual tests that need
    isolation must use the ``fake_redis_client`` fixture which flushes the DB
    between tests.
    """
    server = fakeredis.FakeServer()
    return server


@pytest.fixture
def fake_redis_client(fake_redis_server):
    """A fresh fakeredis.FakeRedis client that is flushed before every test.

    Flushing before (not after) the test means that any data left by a
    previously failed test is cleaned up automatically.

    Usage::

        def test_something(fake_redis_client):
            fake_redis_client.set("key", "value")
            assert fake_redis_client.get("key") == "value"
    """
    client = fakeredis.FakeRedis(
        server=fake_redis_server,
        decode_responses=True,
    )
    client.flushall()
    yield client
    # Nothing to teardown — flushall() at the start of the next test handles it.
