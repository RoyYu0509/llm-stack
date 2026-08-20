import asyncio
import pytest
from inferenceLM.request_receiver.request_receiver import RequestReceiver

pending_queue = asyncio.Queue()
request_store = {}


@pytest.mark.asyncio
async def test_same_order_of_requests():
    """
    并发 submit 10 个 request，每个都被正确 tokenize 并进入 queue，
    顺序与 submit 顺序一致。无 race condition。
    """
    receiver = RequestReceiver("gpt2", pending_queue, request_store)
    request_idx = RequestReceiver.index
    request_ids = [f"{request_idx + i}" for i in range(10)]

    # Submit 10 requests concurrently
    coros = [receiver.submit_request(f"Prompt {i}", f"user_{i}") for i in range(10)]
    results = await asyncio.gather(*coros)
    
    # Check the order should be FIFO
    for i in range(len(request_ids)):
        tokenized_data = await receiver.get_from_request_queue()  # Get the next tokenized data from the queue
        assert tokenized_data.request_id == request_ids[i], f"Expected {request_ids[i]}, got {tokenized_data.request_id}"


@pytest.mark.asyncio
async def test_token_match():
    """
    并发 submit 10 个 request, 每一个 request 都被正确的 tokenize 了, 
    token 的内容与 tokenizer 的输出一致。
    """
    receiver = RequestReceiver("gpt2", pending_queue, request_store)

    # Submit 10 requests concurrently
    coros = [receiver.submit_request(f"Prompt {i}", f"user_{i}") for i in range(10)]
    await asyncio.gather(*coros)

    # Check the tokens should match tokenizer output
    for i in range(10):
        tokenized_data = await receiver.get_from_request_queue()  # Get the next tokenized data from the queue
        expected_tokens = receiver.tokenizer.encode(f"Prompt {i}")
        assert tokenized_data.tokens == expected_tokens, f"Expected {expected_tokens}, got {tokenized_data.tokens}"


@pytest.mark.asyncio
async def test_request_storage():
    """
    并发 submit 10 个 request, 每个 request 都被正确的存储在 request_store 里.
    """
    receiver = RequestReceiver("gpt2", pending_queue, request_store)
    request_idx = RequestReceiver.index
    request_ids = [f"{request_idx + i}" for i in range(10)]

    # Submit 10 requests concurrently
    coros = [receiver.submit_request(f"Prompt {i}", f"user_{i}") for i in range(10)]
    results = await asyncio.gather(*coros)

    # Check each request_id should be in request_store
    for request_id in request_ids:
        assert request_id in receiver.request_store, f"Request ID {request_id} not found in request_store"
