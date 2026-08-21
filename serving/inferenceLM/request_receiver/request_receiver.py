import asyncio
from time import time
from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.data.request import RequestData
from inferenceLM.request_receiver.tokenizer import Tokenizer
from inferenceLM.data.request_status import RequestStatus
from typing import Dict

class RequestReceiver:
    """
    This class is responsible for: 
        接收 request -> tokenization -> 放到 pending queue 里等 Inference Engine 来拿
    
    Attributes:
        pending_queue (asyncio.Queue): Reference to a shared pending queue with Inference Engine
        request_store (dict): A dictionary to store the original RequestData objects, keyed by request_id.
        tokenizer (Tokenizer): An instance of the Tokenizer class for tokenizing input text.
    """
    pending_queue: asyncio.Queue
    request_store: Dict[str, RequestData]
    tokenizer: Tokenizer
    index = 0 # 给每个 Request class 一个 unique request_id

    def __init__(self, tokenizer_name: str, pending_queue: asyncio.Queue, request_store: Dict[str, RequestData]):
        self.pending_queue = pending_queue
        self.request_store = request_store
        self.tokenizer = Tokenizer(tokenizer_name)

    async def submit_request(self, prompt_text: str, user_id: str, max_length: int = 20, do_sample: bool = False) -> str:
        """
        Create a new RequestData & Submit a new request to the pending queue. (可以 wait for queue has space)

        Args:
            prompt_text (str): The raw text prompt from the user.
            user_id (str): The ID of the user who sent the request.
            max_length (int): The maximum length of the generated text.
            do_sample (bool): Whether to use sampling for generation.

        Returns:
            str: The unique request ID assigned to the submitted request.
        """
        # 在接收到 user 的 input text 之后, 在function内部重新包装成一个 RequestData class
        request_data = RequestData(
            prompt_text=prompt_text, 
            user_id=user_id, 
            request_id=str(RequestReceiver.index), 
            timestamp=time(), 
            status=RequestStatus.PENDING,
            max_token_length=max_length,
            do_sample=do_sample
        )
        self.request_store[request_data.request_id] = request_data # 存 request_id : request_data
        RequestReceiver.index += 1
        
        # Create tokenized data
        tokenized_prompt = self.tokenizer.encode(request_data.prompt_text)
        tokenized_data = TokenizedData(request_id=request_data.request_id, tokens=tokenized_prompt)
        
        await self.pending_queue.put(tokenized_data)
        return request_data.request_id

    async def get_from_request_queue(self) -> TokenizedData:
        """
        Get the next tokenized data from the pending queue. (不需要 wait, 因为 Inference Engine 是在等 queue 里有东西了才来拿的)
        
        Returns:
            TokenizedData: The next tokenized data from the pending queue.
        """
        return await self.pending_queue.get()
