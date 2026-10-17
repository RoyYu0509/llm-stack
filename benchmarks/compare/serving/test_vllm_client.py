"""VLLMClient tests: a fake server that streams the chunk shapes vLLM 0.31.0 produces from `/v1/completions`
(completion/serving.py: running `usage` only when stream_options asks for it, `text=""` for tokens the incremental
detokenizer cannot print yet, a final usage-only chunk, then [DONE]). CPU only; the real server is not needed.
"""
import asyncio
import csv
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import closed_loop as cl  # noqa: E402
from vllm_client import VLLMClient  # noqa: E402

pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

TOKEN_S = 0.001


class FakeVLLM:
    """Runs in its own thread/loop so `cl.main` (which calls asyncio.run itself) can talk to it too."""

    def __init__(self, empty_every=0, per_chunk=1, honour_usage=True, status=None, error_after=None):
        self.empty_every, self.per_chunk, self.honour_usage = empty_every, per_chunk, honour_usage
        self.status, self.error_after = status, error_after
        self.bodies = []

    async def _completions(self, request):
        body = await request.json()
        self.bodies.append(body)
        if self.status:
            return web.json_response({"error": {"message": "This model's maximum context length is 1024 tokens"}},
                                     status=self.status)
        n, prompt_n = body["max_tokens"], len(body["prompt"])
        so = body.get("stream_options") or {}
        usage_on = self.honour_usage and so.get("include_usage") and so.get("continuous_usage_stats")
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def send(obj):
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        sent = 0
        while sent < n:
            k = min(self.per_chunk, n - sent)
            sent += k
            if self.error_after is not None and sent > self.error_after:
                await send({"error": {"message": "engine died", "type": "InternalServerError"}})
                break
            text = "" if (self.empty_every and sent % self.empty_every == 0) else "x" * k
            chunk = {"id": "cmpl-1", "object": "text_completion", "model": body["model"],
                     "choices": [{"index": 0, "text": text, "logprobs": None,
                                  "finish_reason": "length" if sent == n else None}]}
            if usage_on:
                chunk["usage"] = {"prompt_tokens": prompt_n, "completion_tokens": sent, "total_tokens": prompt_n + sent}
            await send(chunk)
            await asyncio.sleep(TOKEN_S)
        else:
            if so.get("include_usage") and self.honour_usage:   # the trailing usage-only chunk
                await send({"id": "cmpl-1", "object": "text_completion", "model": body["model"], "choices": [],
                            "usage": {"prompt_tokens": prompt_n, "completion_tokens": n, "total_tokens": prompt_n + n}})
        await resp.write(b"data: [DONE]\n\n")
        return resp

    def __enter__(self):
        ready = threading.Event()

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            app = web.Application()
            app.router.add_post("/v1/completions", self._completions)
            self.runner = web.AppRunner(app)
            self.loop.run_until_complete(self.runner.setup())
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            self.loop.run_until_complete(site.start())
            self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
            ready.set()
            self.loop.run_forever()
            self.loop.run_until_complete(self.runner.cleanup())

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert ready.wait(10), "fake server did not start"
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


def _closed_loop(client_factory, url, n_req=4, new_tokens=12, conc=2):
    async def go():
        client = client_factory(url)
        try:
            return await cl.run_closed_loop(client, [[5, 6, 7]] * n_req, new_tokens=new_tokens, concurrency=conc)
        finally:
            await client.close()

    return asyncio.run(go())


def test_every_token_is_counted_even_when_vllm_sends_empty_text():
    with FakeVLLM(empty_every=3) as srv:
        traces, _ = _closed_loop(VLLMClient, srv.url)
    assert len(traces) == 4 and all(t.ok and len(t.token_times) == 12 for t in traces), (
        "a chunk with text='' still carries a token: the running usage count must be what we count")


def test_the_old_openai_branch_would_have_failed_these_requests():
    """Documents why VLLMClient exists: SGLangClient(endpoint='openai') counts non-empty text chunks."""
    with FakeVLLM(empty_every=3) as srv:
        traces, _ = _closed_loop(lambda u: cl.SGLangClient(u, "openai"), srv.url)
    assert all(not t.ok for t in traces), "if this passes the old branch is fine and VLLMClient is not needed"


def test_request_body_is_greedy_token_ids_with_eos_ignored_and_running_usage():
    with FakeVLLM() as srv:
        _closed_loop(lambda u: VLLMClient(u, model="served-x"), srv.url, n_req=1, new_tokens=7, conc=1)
    b = srv.bodies[0]
    assert b["prompt"] == [5, 6, 7] and b["max_tokens"] == 7 and b["model"] == "served-x"
    assert b["temperature"] == 0.0 and b["ignore_eos"] is True and b["stream"] is True
    assert b["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}


def test_two_tokens_in_one_chunk_become_two_trace_entries():
    with FakeVLLM(per_chunk=2) as srv:
        traces, _ = _closed_loop(VLLMClient, srv.url, new_tokens=10)
    assert all(t.ok and len(t.token_times) == 10 for t in traces)


def test_server_without_running_usage_fails_loudly_instead_of_timestamping_everything_at_the_end(capsys):
    with FakeVLLM(honour_usage=False) as srv:
        traces, _ = _closed_loop(VLLMClient, srv.url)
    assert all(not t.ok for t in traces)
    assert "continuous_usage_stats" in capsys.readouterr().err


def test_http_error_is_reported_with_the_server_message_and_the_request_is_failed(capsys):
    with FakeVLLM(status=400) as srv:
        traces, _ = _closed_loop(VLLMClient, srv.url)
    assert all(not t.ok for t in traces)
    err = capsys.readouterr().err
    assert "HTTP 400" in err and "maximum context length" in err
    assert err.count("first failure") == 1, "only the first failure of a client is printed"


def test_error_event_in_the_stream_fails_the_request(capsys):
    with FakeVLLM(error_after=3) as srv:
        traces, _ = _closed_loop(VLLMClient, srv.url, n_req=2, conc=1)
    assert all(not t.ok for t in traces) and "engine died" in capsys.readouterr().err


def test_run_subcommand_with_backend_vllm_writes_rows_and_passes_the_served_model_name(tmp_path):
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"prompts": [[1, 2, 3]] * 6, "new_tokens": 8, "new_tokens_list": [4, 6, 8, 4, 6, 8]}))
    out = tmp_path / "serving.csv"
    with FakeVLLM(empty_every=2) as srv:
        rc = cl.main(["run", "--backend", "vllm", "--system", "vllm_test_bfloat16", "--url", srv.url,
                      "--served-model-name", "my-model", "--workload", str(wl), "--concurrency", "2",
                      "--reps", "1", "--warmup", "0", "--out", str(out)])
    assert rc == 0 and {b["model"] for b in srv.bodies} == {"my-model"}
    rows = list(csv.DictReader(open(out)))
    assert {r["system"] for r in rows} == {"vllm_test_bfloat16"}
    assert any(r["metric"] == "output_tok_per_s" and float(r["value"]) > 0 for r in rows)
    assert not any(r["metric"] == "n_failed_requests" for r in rows), "no request may be marked failed"
