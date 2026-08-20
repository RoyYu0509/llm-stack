from transformers import GPT2LMHeadModel, GPT2Tokenizer

from inferenceLM.data.request_status import RequestStatus
from inferenceLM.engine.inference_engine import InferenceEngine
from inferenceLM.data.request import RequestData
from inferenceLM.data.tokenized_data import TokenizedData
import asyncio
import pytest
import asyncio
import pytest
from inferenceLM.request_receiver.request_receiver import RequestReceiver

@pytest.fixture(scope="module")
def model():
    return GPT2LMHeadModel.from_pretrained("gpt2")


@pytest.mark.asyncio
async def test_concurrent_multiple_user_data_flow(model: GPT2LMHeadModel):
    pending_queue = asyncio.Queue()
    request_store = {}
    receiver = RequestReceiver("gpt2", pending_queue, request_store)
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    # 3 Users submit requests concurrently for 3 times
    inputs = []
    request_ids = []
    for i in range(3):
        user_input = [{"prompt_text": f"Prompt {i}", "user_id": f"user_{j}"} for j in range(3)]
        inputs.extend(user_input)
        coros = [receiver.submit_request(user_input[j]["prompt_text"], user_input[j]["user_id"]) for j in range(3)]
        request_ids.extend(await asyncio.gather(*coros))

    # Start the Inference Engine to fetching requests and put them in waiting_queue
    task = inference_engine.run() # 在 background register task
    await pending_queue.join()
    task.cancel() # 取消 background task, 已经全部处理完了

    # 1. 所有 request 都在 request_store 里
    for i in range(9):
        assert request_ids[i] in request_store, f"Request ID {request_ids[i]} should be in request_store"

    # 2. 所有 9 个 request status == DONE
    for i in range(9):
        assert request_store[request_ids[i]].status == RequestStatus.DONE, f"Request ID {request_ids[i]} should be marked as DONE"  
        
    # 3. 所有 9 个 request 都有 generated tokens
    for i in range(9):
        assert len(request_store[request_ids[i]].generated_tokens) > 0, f"Request ID {request_ids[i]} should have generated tokens"

    # 4. 每个 request 的 user_id 和 prompt_text 与提交时一致
    for req_id, input in zip(request_ids, inputs):
        assert request_store[req_id].user_id == input["user_id"], f"User ID for Request ID {req_id} should match"
        assert request_store[req_id].prompt_text == input["prompt_text"], f"Prompt text for Request ID {req_id} should match"


@pytest.mark.asyncio
async def test_inference_running_while_input_new_requests(model: GPT2LMHeadModel):
    pending_queue = asyncio.Queue()
    request_store = {}
    receiver = RequestReceiver("gpt2", pending_queue, request_store)
    inference_engine = InferenceEngine(model, pending_queue, request_store)

    # 开启 inference engine
    task = inference_engine.run() # 在 background register task

    request_ids = []

    # receiver 开始接收新 request
    for i in range(3):
        user_input = f"Prompt {i}"
        request_id = await receiver.submit_request(user_input, f"user_{i}")
        request_ids.append(request_id)
        await asyncio.sleep(0.5) # 等半秒再提交下一个 request, 模拟用户输入的间隔

    # 等待所有 request 都 inference engine 拿走, 然后处理完
    await pending_queue.join() 
    task.cancel() # 取消 background task, 已经全部处理完了

    # check 所有 request 都被处理了
    for i in range(3):
        req_id = request_ids[i] # 不能用 str(i) 因为 request_id 是 static method 生成的, 不是简单的数字字符串
        assert req_id in request_store, f"Request ID {req_id} should be in request_store"
        assert request_store[req_id].status == RequestStatus.DONE, f"Request ID {req_id} should be marked as DONE"  
        assert len(request_store[req_id].generated_tokens) > 0, f"Request ID {req_id} should have generated tokens"
        assert request_store[req_id].prompt_text == f"Prompt {i}", f"Prompt text for Request ID {req_id} should match"
        assert request_store[req_id].user_id == f"user_{i}", f"User ID for Request ID {req_id} should match"    
