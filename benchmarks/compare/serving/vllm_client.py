"""vLLM client for the closed-loop harness (I1): the same `generate()` interface as SGLangClient.

Why this is not just `SGLangClient(endpoint="openai")`: that branch counts SSE chunks whose `text` is non-empty. With
ignore_eos and random-init weights the model emits arbitrary token ids, many of them partial UTF-8 byte pieces; vLLM's
incremental detokenizer holds those back and sends `text=""`. Counting chunks then under-counts, and `run_closed_loop`
marks the request failed ("fewer tokens than requested"). This client asks vLLM for a running token count in every
chunk (`stream_options.continuous_usage_stats`, checked in vllm 0.31.0 completion/serving.py) and yields one item per
new token, exactly like SGLang's native endpoint does with `meta_info.completion_tokens`.

If a content chunk arrives without that running count the client raises instead of guessing: with only the final
usage chunk every token would get the end-of-stream timestamp and TTFT / TPOT would be wrong without any error.

"""
from __future__ import annotations

import json
import sys
from typing import AsyncIterator, Sequence


class VLLMClient:
    """POST {url}/v1/completions with a token-id prompt, greedy, EOS ignored, streaming."""

    def __init__(self, url: str, model: str = "default"):
        self.url, self.model = url.rstrip("/"), model
        self._session = None
        self._reported = False

    async def _sess(self):
        import aiohttp

        if self._session is None:
            # limit=0: no cap (aiohttp's default of 100 connections would turn client-side waiting into server TTFT at C > 100)
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                                  timeout=aiohttp.ClientTimeout(total=3600))
        return self._session

    def body(self, prompt_ids: Sequence[int], new_tokens: int) -> dict:
        return {"model": self.model, "prompt": list(prompt_ids), "max_tokens": int(new_tokens),
                "temperature": 0.0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}

    def _report(self, msg: str) -> None:
        """run_closed_loop turns any exception into status=failed; print the first cause so it is not lost."""
        if not self._reported:
            self._reported = True
            print(f"[vllm_client] first failure: {msg}", file=sys.stderr, flush=True)

    async def generate(self, prompt_ids: Sequence[int], new_tokens: int) -> AsyncIterator[int]:
        s = await self._sess()
        seen = 0
        async with s.post(f"{self.url}/v1/completions", json=self.body(prompt_ids, new_tokens)) as resp:
            if resp.status >= 400:
                msg = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                self._report(msg)
                raise RuntimeError(msg)
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    msg = f"error event in stream: {str(obj['error'])[:300]}"
                    self._report(msg)
                    raise RuntimeError(msg)
                usage = obj.get("usage")
                if obj.get("choices") and not usage:
                    msg = "content chunk without a running usage count (continuous_usage_stats not honoured)"
                    self._report(msg)
                    raise RuntimeError(msg)
                if usage:
                    n = int(usage["completion_tokens"])
                    for _ in range(max(0, n - seen)):
                        yield 0
                    seen = max(seen, n)

    async def close(self):
        if self._session is not None:
            await self._session.close()
