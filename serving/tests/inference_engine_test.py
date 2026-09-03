from transformers import GPT2LMHeadModel, GPT2Tokenizer

from inferenceLM.data.request_status import RequestStatus
from inferenceLM.engine.inference_engine import InferenceEngine
from inferenceLM.data.request import RequestData
from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.engine.inference_engine_status import InferenceEngineStatus
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

    assert inference_engine.status == InferenceEngineStatus.INIT, "Inference Engine should be initialized as INIT status"
    ""

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
    await inference_engine.shutdown(draining=True)
    assert task.done(), "Inference Engine task should have completed after draining shutdown"
    
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
    await inference_engine.shutdown(draining=True)
    assert task.done(), "Inference Engine task should have completed after draining shutdown"

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
        await inference_engine.shutdown(draining=True)
        assert task.done(), "Inference Engine task should have completed after draining shutdown"

    assert inference_engine.request_store[request_id].status == RequestStatus.FAILED, f"Request ID {request_id} should be marked as FAILED due to LM engine exception"
    assert len(inference_engine.request_store[request_id].generated_tokens) == 0, f"Request ID {request_id} should not have generated tokens due to LM engine exception"


@pytest.mark.asyncio
async def test_inference_engine_shutdown_on_empty_queue_returns_promptly(model: GPT2LMHeadModel):
    # AC: kill() 原本的洞是——空队列上 engine 卡在 pending_queue.get() 上,永远不返回。
    # shutdown(drain=True) 必须能在有限时间内正常返回,不是"改完感觉应该好了"。
    pending_queue = asyncio.Queue()
    request_store = {}
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    task = inference_engine.run()  # 队列是空的,后台循环这时应该正卡在 pending_queue.get() 上

    # 用 wait_for 给一个明确的上限:真的卡死会在这里超时报错,而不是让整个测试套件挂住
    await asyncio.wait_for(inference_engine.shutdown(draining=True), timeout=5)

    assert task.done(), "Inference Engine task should have completed after draining shutdown on an empty queue"
    assert inference_engine.status == InferenceEngineStatus.STOPPED, "Inference Engine should be in STOPPED state after draining shutdown"


@pytest.mark.asyncio
async def test_inference_engine_immediate_draining_shutdown(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
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

    # Start the inference engine 然后把 LMEngine.inference 这个异步方法临时替换（Mock)
    async def _slow_inference(*args, **kwargs):
        await asyncio.sleep(5)
        return [0]  # Return a dummy token

    with patch(target = "inferenceLM.engine.inference_engine.LMEngine.inference", new=AsyncMock(side_effect= _slow_inference)):  # Simulate a long-running inference
        task = inference_engine.run() # create 一个 background task 然后直接 return 回来
        # Immediately trigger draining-shutdown before the inference can complete
        await inference_engine.shutdown(draining=True)  

    assert task.done(), "Inference Engine task should be completed after draining shutdown"
    assert inference_engine.status == InferenceEngineStatus.STOPPED, "Inference Engine should be in STOPPED state after draining shutdown"
    assert inference_engine.request_store[request_id].status == RequestStatus.DONE, f"Request ID {request_id} should be marked as DONE due to draining shutdown"

@pytest.mark.asyncio
async def test_inference_engine_immediate_nondraining_shutdown(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
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

    # Start the inference engine 然后把 LMEngine.inference 这个异步方法临时替换（Mock)
    async def _slow_inference(*args, **kwargs):
        await asyncio.sleep(5)
        return [0]  # Return a dummy token

    with patch(target = "inferenceLM.engine.inference_engine.LMEngine.inference", new=AsyncMock(side_effect= _slow_inference)):  # Simulate a long-running inference
        task = inference_engine.run() # create 一个 background task 然后直接 return 回来
        # Immediately trigger draining-shutdown before the inference can complete
        await inference_engine.shutdown(draining=False)  

    assert task.done(), "Task should be cancelled after immediate shutdown"
    assert inference_engine.status == InferenceEngineStatus.STOPPED, "Inference Engine should be in STOPPED state after draining shutdown"
    assert inference_engine.request_store[request_id].status == RequestStatus.FAILED, f"Request ID {request_id} should be marked as FAILED due to draining shutdown"

@pytest.mark.asyncio
async def test_inference_engine_draining_shutdown_while_processing(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
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

    # Start the inference engine 然后把 LMEngine.inference 这个异步方法临时替换（Mock)
    async def _slow_inference(*args, **kwargs):
        await asyncio.sleep(5)
        return [0]  # Return a dummy token

    with patch(target = "inferenceLM.engine.inference_engine.LMEngine.inference", new=AsyncMock(side_effect= _slow_inference)):  # Simulate a long-running inference
        task = inference_engine.run() # create 一个 background task 然后直接 return 回来
        while inference_engine.request_store[request_id].status != RequestStatus.PROCESSING:
            await asyncio.sleep(0.1)  # Wait until the request is being processed
        await inference_engine.shutdown(draining=True)  # Trigger draining shutdown while processing

    assert task.done(), "Task should be completed after draining shutdown"
    assert inference_engine.status == InferenceEngineStatus.STOPPED, "Inference Engine should be in STOPPED state after draining shutdown"
    assert inference_engine.request_store[request_id].status == RequestStatus.DONE, f"Request ID {request_id} should be marked as DONE due to draining shutdown"

@pytest.mark.asyncio
async def test_inference_engine_nondraining_shutdown_while_processing(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
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

    # Start the inference engine 然后把 LMEngine.inference 这个异步方法临时替换（Mock)
    async def _slow_inference(*args, **kwargs):
        await asyncio.sleep(5)
        return [0]  # Return a dummy token

    with patch(target = "inferenceLM.engine.inference_engine.LMEngine.inference", new=AsyncMock(side_effect= _slow_inference)):  # Simulate a long-running inference
        task = inference_engine.run() # create 一个 background task 然后直接 return 回来
        while inference_engine.request_store[request_id].status != RequestStatus.PROCESSING:
            await asyncio.sleep(0.1)  # Wait until the request is being processed
        await inference_engine.shutdown(draining=False)  # Trigger non-draining shutdown while processing

    assert task.done(), "Task should be cancelled after non-draining shutdown"
    assert inference_engine.status == InferenceEngineStatus.STOPPED, "Inference Engine should be in STOPPED state after non-draining shutdown"
    assert inference_engine.request_store[request_id].status == RequestStatus.FAILED, f"Request ID {request_id} should be marked as FAILED due to non-draining shutdown"

@pytest.mark.asyncio
async def test_inference_engine_run_must_be_in_init_status(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    pending_queue = asyncio.Queue()
    request_store = {}
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    # Manually set the status to RUNNING to simulate an invalid state
    inference_engine.status = InferenceEngineStatus.RUNNING
    with pytest.raises(RuntimeError, match="Inference Engine must be in INIT status to run, Not RUNNING"):
        inference_engine.run()
    inference_engine.status = InferenceEngineStatus.STOPPED
    with pytest.raises(RuntimeError, match="Inference Engine must be in INIT status to run, Not STOPPED"):
        inference_engine.run()
    inference_engine.status = InferenceEngineStatus.DRAINING
    with pytest.raises(RuntimeError, match="Inference Engine must be in INIT status to run, Not DRAINING"):
        inference_engine.run()

@pytest.mark.asyncio
async def test_shutdown_failed_if_not_in_running_status(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    pending_queue = asyncio.Queue()
    request_store = {}
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    # Manually set the status to INIT to simulate an invalid state for shutdown
    inference_engine.status = InferenceEngineStatus.INIT
    with pytest.raises(RuntimeError , match="Inference Engine must be in RUNNING status to shutdown, Not INIT"):
        await inference_engine.shutdown(draining=False)
    with pytest.raises(RuntimeError , match="Inference Engine must be in RUNNING status to shutdown, Not INIT"):
        await inference_engine.shutdown(draining=True)
    inference_engine.status = InferenceEngineStatus.STOPPED
    with pytest.raises(RuntimeError, match="Inference Engine must be in RUNNING status to shutdown, Not STOPPED"):
        await inference_engine.shutdown(draining=False)
    with pytest.raises(RuntimeError, match="Inference Engine must be in RUNNING status to shutdown, Not STOPPED"):
        await inference_engine.shutdown(draining=True)

    inference_engine.status = InferenceEngineStatus.DRAINING
    with pytest.raises(RuntimeError, match="Inference Engine must be in RUNNING status to shutdown, Not DRAINING"):
        await inference_engine.shutdown(draining=False)
    with pytest.raises(RuntimeError, match="Inference Engine must be in RUNNING status to shutdown, Not DRAINING"):
        await inference_engine.shutdown(draining=True)
