from transformers import GPT2LMHeadModel, GPT2Tokenizer

from inferenceLM.data.request_status import RequestStatus
from inferenceLM.engine.inference_engine import InferenceEngine
from inferenceLM.data.request import RequestData
from inferenceLM.data.tokenized_data import TokenizedData
import asyncio
import pytest

from unittest.mock import patch
from unittest.mock import AsyncMock


@pytest.fixture(scope="module")
def model():
    return GPT2LMHeadModel.from_pretrained("gpt2")

@pytest.mark.asyncio
async def test_inference_engine_is_not_open(model: GPT2LMHeadModel):
    pending_queue = asyncio.Queue()
    request_store = {}
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    assert not inference_engine.open, "Inference Engine should be initialized as closed"

@pytest.mark.asyncio
async def test_inference_engine_async_fetch_all_requests(model: GPT2LMHeadModel):
    pending_queue = asyncio.Queue()
    request_store = {}

    # create 10 tokenized requests and put them in the pending_queue
    for i in range(10):
        user_input = f"Prompt {i}"
        input_ids = [i]  # dummy tokenized data, just use the index as the token for simplicity
        request_id = f"request_{i}"
        request_store[request_id] = RequestData(
            request_id=request_id,
            timestamp=i, 
            user_id=f"user_{i}",
            prompt_text=user_input,
            status=RequestStatus.PENDING,
            generated_tokens=[],
            max_token_length=200,
        )
        await pending_queue.put(TokenizedData(request_id=request_id, tokens=input_ids))
    
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    # Check all requests are in the inference engine's request_store
    for i in range(10):
        assert f"request_{i}" in inference_engine.request_store, f"Request ID request_{i} should be in request_store"
        assert inference_engine.request_store[f"request_{i}"].status == RequestStatus.PENDING, f"Request ID request_{i} should be in PENDING status"

    # Start the Inference Engine to fetching requests and put them in waiting_queue
    task = inference_engine.run() # 不要等 run(), 它是一个background task, 直接继续往下走到 sync point 再自然等这个task结束
    await pending_queue.join()
    task.cancel() 
    
    # check all request is marked as DONE
    for i in range(10):
        assert inference_engine.request_store[f"request_{i}"].status == RequestStatus.DONE, f"Request ID request_{i} should be processed and marked as DONE"
        assert len(inference_engine.request_store[f"request_{i}"].generated_tokens) > 0, f"Request ID request_{i} should have generated tokens"


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

@pytest.mark.asyncio
async def test_inference_engine_reject_prompt_equals_max_len(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    pending_queue = asyncio.Queue()
    request_store = {}

    user_input = "Hello, how are you? My name is Yifan and "
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids[:, :10] # truncate the input to max_length
    request_id = "test_request"
    request_store[request_id] = RequestData(
        request_id=request_id,
        timestamp=0, 
        user_id="user_1",
        prompt_text=user_input,
        status=RequestStatus.PENDING,
        generated_tokens=[],
        max_token_length=1,
    )
    await pending_queue.put(TokenizedData(request_id=request_id, tokens=input_ids[0].tolist()))

    inference_engine = InferenceEngine(model, pending_queue, request_store)
    
    task = inference_engine.run() 
    await pending_queue.join()  # Wait until the request in pending_queue is processed
    task.cancel()  # Cancel the background task after processing is done

    assert inference_engine.request_store[request_id].status == RequestStatus.FAILED, f"Request ID {request_id} should be marked as FAILED due to prompt length equals max_length"
    assert len(inference_engine.request_store[request_id].generated_tokens) == 0, f"Request ID {request_id} should not have generated tokens due to prompt length equals max_length"


@pytest.mark.asyncio
async def test_inference_engine_handle_lm_engine_exception(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    pending_queue = asyncio.Queue()
    request_store = {}
    inference_engine = InferenceEngine(model, pending_queue, request_store)
    # Add a dummy request to the pending_queue
    user_input = "Hello, how are you? My name is Yifan and "
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids
    request_id = "test_request"
    request_store[request_id] = RequestData(
        request_id=request_id,
        timestamp=0,
        user_id="user_1",
        prompt_text=user_input,
        status=RequestStatus.PENDING,
        generated_tokens=[],
        max_token_length=200,
    )
    await pending_queue.put(TokenizedData(request_id=request_id, tokens=input_ids[0].tolist()))

    runtime_error_inference = AsyncMock(side_effect=RuntimeError("Inference failed due to some error"))
    # `patch` 需要完整的 import path
    with patch(target = "inferenceLM.engine.inference_engine.LMEngine.inference", new=runtime_error_inference):
        task = inference_engine.run()
        await pending_queue.join()  # Wait until the request in pending_queue is processed
        task.cancel()  # Cancel the background task after processing is done

    assert inference_engine.request_store[request_id].status == RequestStatus.FAILED, f"Request ID {request_id} should be marked as FAILED due to LM engine exception"
    assert len(inference_engine.request_store[request_id].generated_tokens) == 0, f"Request ID {request_id} should not have generated tokens due to LM engine exception"
