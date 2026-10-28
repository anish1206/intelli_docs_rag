import asyncio
import math
import os
import ssl

import httpx
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError


def _positive_seconds(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class LlmRequestError(Exception):
    def __init__(self, timed_out: bool = False) -> None:
        super().__init__("LLM request timed out" if timed_out else "LLM request failed")
        self.timed_out = timed_out


class BoundedOllama:
    """Invoke Ollama with a cancellable deadline and bounded transient retries."""

    def __init__(self, model: str, base_url: str, temperature: float) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.connect_timeout = _positive_seconds("OLLAMA_CONNECT_TIMEOUT", "10")
        self.read_timeout = _positive_seconds("OLLAMA_READ_TIMEOUT", "60")
        self.request_timeout = _positive_seconds("OLLAMA_REQUEST_TIMEOUT", "120")
        self.backoff = _positive_seconds("OLLAMA_RETRY_BACKOFF", "0.5")
        self.retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
        if not 0 <= self.retries <= 10:
            raise ValueError("OLLAMA_MAX_RETRIES must be between 0 and 10")

    def invoke(self, prompt: str) -> BaseMessage:
        return asyncio.run(self._invoke(prompt))

    async def _invoke(self, prompt: str) -> BaseMessage:
        timeout = httpx.Timeout(self.read_timeout, connect=self.connect_timeout)
        attempt = 0
        while True:
            try:
                async with httpx.AsyncHTTPTransport() as transport:
                    client = ChatOllama(
                        model=self.model,
                        base_url=self.base_url,
                        temperature=self.temperature,
                        client_kwargs={"timeout": timeout},
                        async_client_kwargs={"transport": transport},
                    )
                    return await asyncio.wait_for(
                        client.ainvoke(prompt), timeout=self.request_timeout
                    )
            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                failure = LlmRequestError(timed_out=True)
                failure.__cause__ = exc
                transient = True
            except httpx.ConnectError as exc:
                cause = exc.__cause__
                while cause is not None and not isinstance(cause, ssl.SSLError):
                    cause = cause.__cause__
                failure = LlmRequestError()
                failure.__cause__ = exc
                transient = cause is None
            except httpx.RemoteProtocolError as exc:
                failure = LlmRequestError()
                failure.__cause__ = exc
                transient = True
            except ResponseError as exc:
                failure = LlmRequestError(timed_out=exc.status_code == 408)
                failure.__cause__ = exc
                transient = exc.status_code in (408, 429, 500, 502, 503, 504)
            except httpx.RequestError as exc:
                raise LlmRequestError() from exc
            if not transient or attempt == self.retries:
                raise failure
            await asyncio.sleep(self.backoff * (2**attempt))
            attempt += 1
