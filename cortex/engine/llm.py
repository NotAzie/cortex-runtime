"""
Unified asynchronous streaming client for local inference servers.
Direct HTTP keep-alive connection pool targeting llama.cpp server or Ollama daemon.
"""

import aiohttp
import json
import time
from typing import AsyncGenerator, Dict, Any, Optional
from cortex.types import TokenChunk
from cortex.exceptions import InferenceEngineError


class StreamingLLMClient:
    def __init__(self, endpoint_url: str = "http://127.0.0.1:8080/completion", timeout_sec: float = 20.0):
        self.endpoint_url = endpoint_url
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def stream_generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 128) -> AsyncGenerator[TokenChunk, None]:
        session = await self._get_session()
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["<|end|>", "<|user|>"]
        }

        try:
            async with session.post(self.endpoint_url, json=payload) as resp:
                if resp.status != 200:
                    raise InferenceEngineError(f"Inference HTTP error {resp.status}: {await resp.text()}")

                start_tok = time.perf_counter()
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    if line_str.startswith("data: "):
                        line_str = line_str[6:].strip()
                    
                    try:
                        data = json.loads(line_str)
                        content = data.get("content", "")
                        stop = data.get("stop", False)
                        delta_ms = (time.perf_counter() - start_tok) * 1000.0
                        start_tok = time.perf_counter()

                        yield TokenChunk(
                            text=content,
                            finish_reason="stop" if stop else None,
                            latency_ms=round(delta_ms, 2)
                        )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            if not isinstance(e, InferenceEngineError):
                raise InferenceEngineError(f"Connection failure to {self.endpoint_url}: {e}") from e
            raise

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
