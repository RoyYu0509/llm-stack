"""End-to-end: a locally-trained TransformerLM served through the real request path.

This is the seam between the two projects -- `RequestReceiver` -> `pending_queue` ->
`InferenceEngine` -> `LMEngine` -> `TransformerLMAdapter` -> our own pretrained weights,
with no HuggingFace model anywhere in the chain except the tokenizer (GPT-2's BPE, which
is what the model was trained on).

It is skipped unless both halves are actually present, because neither is part of this
repo's normal test environment:

    TRAINED_LM_CKPT=/path/to/checkpoint_epoch_0100.pt \
    TRAINED_LM_CONFIG=/path/to/real_pretrain_config.json \
    uv run pytest tests/trained_model_integration_test.py -v -s

`cs336_basics` (the training project's package) must also be importable. It is
deliberately NOT a hard dependency of `inferenceLM`: the adapter takes an already-built
module, so only this test needs the training package, and the two repos stay decoupled.
"""

import asyncio
import json
import os

import pytest
import torch

from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.engine.inference_engine import InferenceEngine
from inferenceLM.engine.lm_engine import LMEngine
from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter

CKPT = os.environ.get("TRAINED_LM_CKPT")
CONFIG = os.environ.get("TRAINED_LM_CONFIG")

cs336 = pytest.importorskip("cs336_basics.lm", reason="training package not installed")

pytestmark = pytest.mark.skipif(
    not (CKPT and CONFIG and os.path.exists(CKPT) and os.path.exists(CONFIG)),
    reason="set TRAINED_LM_CKPT and TRAINED_LM_CONFIG to run the trained-model integration test",
)


@pytest.fixture(scope="module")
def adapted_model():
    from cs336_basics.lm import TransformerLM

    cfg = json.load(open(CONFIG))
    m = cfg["model"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = TransformerLM(
        vocab_size=m["vocab_size"], context_length=m["context_length"],
        num_layers=m["num_layers"], d_model=m["d_model"], heads_num=m["num_heads"],
        d_ff=m["d_ff"], theta=m.get("rope_theta", 10_000.0),
        device=device, dtype=torch.float32,
    )
    ckpt = torch.load(CKPT, map_location=device)
    # Training runs with torch.compile, which prefixes every state_dict key.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    return TransformerLMAdapter(
        model,
        context_length=m["context_length"],
        vocab_size=m["vocab_size"],
    )


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import GPT2Tokenizer

    # Token-for-token identical to the tiktoken "gpt2" encoding used for training.
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.mark.asyncio
async def test_lm_engine_generates_from_trained_weights(adapted_model, tokenizer):
    prompt = "The history of the internet began in the"
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    tokenized = TokenizedData(request_id="trained_lm", tokens=tokens)

    generated = await LMEngine(adapted_model).inference(
        tokenized, do_sample=False, max_length=len(tokens) + 24
    )

    assert len(generated) > 0
    assert all(0 <= t < adapted_model.config.vocab_size for t in generated)
    print(f"\nprompt:    {prompt}\ncontinued: {tokenizer.decode(generated)}")


@pytest.mark.asyncio
async def test_full_request_path_through_inference_engine(adapted_model, tokenizer):
    """The whole pipeline, not just LMEngine: a request goes on the queue and comes back
    out of request_store marked DONE with tokens attached."""
    import time

    from inferenceLM.data.request import RequestData

    pending_queue: asyncio.Queue = asyncio.Queue()
    request_store: dict = {}

    prompt = "Paris is the capital and most populous city of"
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    request_id = "e2e-request"
    request_store[request_id] = RequestData(
        user_id="test-user",
        request_id=request_id,
        timestamp=time.time(),
        status=RequestStatus.PENDING,
        prompt_text=prompt,
        max_token_length=len(tokens) + 24,
    )
    await pending_queue.put(TokenizedData(request_id=request_id, tokens=tokens))

    engine = InferenceEngine(
        model=adapted_model,
        pending_queue=pending_queue,
        request_store=request_store,
    )
    task = engine.run()
    await pending_queue.join()
    await engine.shutdown(draining=True)
    assert task.done(), "Inference Engine task should have completed after draining shutdown"

    done = request_store[request_id]
    assert done.status == RequestStatus.DONE
    assert len(done.generated_tokens) > 0
    print(f"\nprompt:    {prompt}\ncontinued: {tokenizer.decode(done.generated_tokens)}")
