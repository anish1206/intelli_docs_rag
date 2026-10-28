import asyncio
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from ollama import ResponseError

from backend.llm_client import BoundedOllama, LlmRequestError


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("OLLAMA_MAX_RETRIES", "2")
    monkeypatch.setenv("OLLAMA_RETRY_BACKOFF", "0.001")
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "0.02")
    return BoundedOllama("model", "http://localhost:11434", 0)


def test_transient_failure_retries_then_recovers(llm):
    invoke = AsyncMock(
        side_effect=[httpx.ConnectError("unavailable"), AIMessage(content="answer")]
    )
    with patch("backend.llm_client.ChatOllama") as factory:
        factory.return_value.ainvoke = invoke
        assert llm.invoke("question").content == "answer"
    assert invoke.await_count == 2
    timeout = factory.call_args.kwargs["client_kwargs"]["timeout"]
    assert timeout.connect == llm.connect_timeout
    assert timeout.read == llm.read_timeout


def test_permanent_response_is_not_retried(llm):
    invoke = AsyncMock(side_effect=ResponseError("bad request", status_code=400))
    with patch("backend.llm_client.ChatOllama") as factory:
        factory.return_value.ainvoke = invoke
        with pytest.raises(LlmRequestError) as error:
            llm.invoke("question")
    assert not error.value.timed_out
    assert invoke.await_count == 1


def test_tls_failure_is_not_retried(llm):
    connection_error = httpx.ConnectError("TLS failed")
    connection_error.__cause__ = ssl.SSLCertVerificationError("invalid certificate")
    invoke = AsyncMock(side_effect=connection_error)
    with patch("backend.llm_client.ChatOllama") as factory:
        factory.return_value.ainvoke = invoke
        with pytest.raises(LlmRequestError):
            llm.invoke("question")
    assert invoke.await_count == 1


def test_timeout_cancels_each_attempt_and_stops(llm):
    cancelled = []

    async def stalled(_):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(True)

    invoke = AsyncMock(side_effect=stalled)
    with patch("backend.llm_client.ChatOllama") as factory:
        factory.return_value.ainvoke = invoke
        with pytest.raises(LlmRequestError) as error:
            llm.invoke("question")
    assert error.value.timed_out
    assert invoke.await_count == 3
    assert len(cancelled) == 3


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_deadline_is_rejected(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", value)
    with pytest.raises(ValueError):
        BoundedOllama("model", "http://localhost:11434", 0)


def test_real_ollama_http_retry_and_response(llm):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            status = 503 if len(calls) == 1 else 200
            body = (
                json.dumps({"error": "busy"})
                if status == 503
                else json.dumps(
                    {
                        "model": "model",
                        "message": {"role": "assistant", "content": "answer"},
                        "done": True,
                    }
                )
                + "\n"
            ).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm.base_url = f"http://127.0.0.1:{server.server_port}"
        llm.request_timeout = 2
        assert llm.invoke("question").content == "answer"
        assert calls == ["/api/chat", "/api/chat"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_total_deadline_stops_a_continuously_streaming_response(llm):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            chunk = (
                json.dumps(
                    {
                        "model": "model",
                        "message": {"role": "assistant", "content": "x"},
                        "done": False,
                    }
                )
                + "\n"
            ).encode()
            try:
                for _ in range(100):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm.base_url = f"http://127.0.0.1:{server.server_port}"
        llm.request_timeout = 0.15
        llm.read_timeout = 1
        llm.retries = 0
        started = time.monotonic()
        with pytest.raises(LlmRequestError) as error:
            llm.invoke("question")
        assert error.value.timed_out
        assert time.monotonic() - started < 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
