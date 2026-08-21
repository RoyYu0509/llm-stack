import asyncio

from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData
from typing import Dict
from inferenceLM.data.request import RequestData
import transformers

from inferenceLM.engine.lm_engine import LMEngine

class InferenceEngine:
    """
    This class is responsible for:
        从 pending queue 里拿 tokenized data -> run LM inference  
    
    Attributes:
        lm_engine (LMEngine): An instance of the LMEngine class to perform language model inference.
        pending_queue (asyncio.Queue): Reference to a shared waiting queue with Request Receiver
        request_store (dict): A dictionary to store the original RequestData objects, keyed by request_id.
        open (bool): A flag to indicate whether the Inference Engine is running and should keep fetching requests.
    """
    pending_queue: asyncio.Queue[TokenizedData]
    request_store: Dict[str, RequestData]
    open: bool

    def __init__(
            self, 
            model: transformers.PreTrainedModel, 
            pending_queue: asyncio.Queue, 
            request_store: Dict[str, RequestData]
    ):
        self.lm_engine = LMEngine(model)
        self.pending_queue = pending_queue
        self.request_store = request_store
        self.open = False
    
    async def _keep_get_request_and_inference(self):
        """
        A coroutine that keeps:
            1. getting requests from the pending_queue and
            2. running inference on them.
        """
        while self.open:
            # fetch request
            tokenized_data = await self.pending_queue.get()

            # Do inference & save back to buffer
            self.request_store[tokenized_data.request_id].status = RequestStatus.PROCESSING
            try:
                self.request_store[tokenized_data.request_id].generated_tokens = await self.lm_engine.inference(
                    tokenized_data,
                    max_length=self.request_store[tokenized_data.request_id].max_token_length
                )
                self.request_store[tokenized_data.request_id].status = RequestStatus.DONE
            except Exception as e:
                self.request_store[tokenized_data.request_id].status = RequestStatus.FAILED
                print(f"Request ID {tokenized_data.request_id} failed during inference with error: {str(e)}")
            finally:
                self.pending_queue.task_done()

    def run(self) -> asyncio.Task:
        """
        Start the Inference Engine to keep fetching requests and run inference.
        """
        self.open = True
        return asyncio.create_task(self._keep_get_request_and_inference())

    def kill(self):
        """
        Shut down the Inference Engine and kill any active jobs.
        """
        self.open = False


