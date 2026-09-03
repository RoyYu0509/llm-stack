import asyncio

from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData
from typing import Dict
from inferenceLM.data.request import RequestData
import transformers
from inferenceLM.engine.inference_engine_status import InferenceEngineStatus


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
    pending_queue: asyncio.Queue[TokenizedData|None] # None for shutting down
    request_store: Dict[str, RequestData]
    status: InferenceEngineStatus

    def __init__(
            self, 
            model: transformers.PreTrainedModel, 
            pending_queue: asyncio.Queue[TokenizedData|None], 
            request_store: Dict[str, RequestData]
    ):
        self.lm_engine = LMEngine(model)
        self.pending_queue = pending_queue
        self.request_store = request_store
        self.status = InferenceEngineStatus.INIT
        self.shutdown_event = asyncio.Event()  # Event to signal shutdown completion
    
    async def _keep_get_request_and_inference(self):
        """
        A coroutine that keeps:
            1. getting requests from the pending_queue and
            2. running inference on them.
        """
        while self.status == InferenceEngineStatus.RUNNING or self.status == InferenceEngineStatus.DRAINING:
            # fetch request
            tokenized_data = await self.pending_queue.get()
            # 如果 fetch 到的是 shutdown signal, 就直接 break loop
            if tokenized_data is None:  # Check for shutdown signal 
                self.pending_queue.task_done()  # Mark the dummy task as done
                break  # Exit the loop to stop processing further requests

            # Do inference & save back to buffer
            self.request_store[tokenized_data.request_id].status = RequestStatus.PROCESSING
            try:
                # 两个 tasks: 一个是 inference, 一个是 shutdown_event.wait(), 看谁先完成
                inferencee_task = asyncio.create_task(self.lm_engine.inference(tokenized_data,max_length=self.request_store[tokenized_data.request_id].max_token_length))
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                # await asyncio.sleep(0) # 让出一次线程
                tasks = [inferencee_task,shutdown_task]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # shutdown event 先完成
                if inferencee_task not in done: 
                    # Non-Draining shutdown
                    if self.status == InferenceEngineStatus.STOPPED: 
                        # 先检查是哪种 shutdown, draining == True or False
                        inferencee_task.cancel()
                        try:
                            generated_tokens = await inferencee_task
                        except asyncio.CancelledError:
                            self.request_store[tokenized_data.request_id].status = RequestStatus.FAILED
                    # Draining shutdown
                    elif self.status == InferenceEngineStatus.DRAINING:
                        generated_tokens = await inferencee_task
                        self.request_store[tokenized_data.request_id].generated_tokens = generated_tokens
                        self.request_store[tokenized_data.request_id].status = RequestStatus.DONE
                    # Other cases, just mark as failed
                    else:
                        self.request_store[tokenized_data.request_id].status = RequestStatus.FAILED
                        print(f"Request ID {tokenized_data.request_id} failed with unknown error")
                # inference task 先完成
                else:
                    generated_tokens = await inferencee_task
                    self.request_store[tokenized_data.request_id].generated_tokens = generated_tokens
                    self.request_store[tokenized_data.request_id].status = RequestStatus.DONE
            
            except Exception as e:
                self.request_store[tokenized_data.request_id].status = RequestStatus.FAILED
                print(f"Request ID {tokenized_data.request_id} failed during inference with error: {str(e)}")

            finally:
                self.pending_queue.task_done()

        # while loop 已经退出, engine 已经 stopped, 把 queue 中剩下的 requests 标成 failed 清理掉
        self.clean_queue()

    def run(self) -> asyncio.Task:
        """
        Start the Inference Engine to keep fetching requests and run inference.
        """
        if self.status != InferenceEngineStatus.INIT:
            raise RuntimeError(f"Inference Engine must be in INIT status to run, Not {self.status.name}")
        self.status = InferenceEngineStatus.RUNNING
        return asyncio.create_task(self._keep_get_request_and_inference())

    async def shutdown(self, draining: bool = True):
        """
        Shut down the Inference Engine and kill any active jobs.
        """
        if self.status not in [InferenceEngineStatus.RUNNING]:
            raise RuntimeError(f"Inference Engine must be in RUNNING status to shutdown, Not {self.status.name}")

        self.shutdown_event.set() # 触发 shutdown_event
        self.pending_queue.put_nowait(None)  # Dummy data to unblock the queue
        
        if draining:
            self.status = InferenceEngineStatus.DRAINING
            await self.pending_queue.join()  # 让出线程, 等待 pending_queue 里的所有任务完成
            # Draining 就是等queue中所有task做完之后, 再把 status 改成 STOPPED, 并且 set shutdown_event
            self.status = InferenceEngineStatus.STOPPED
        else:
            # Not draining, 直接停止, 不等 queue 中的任务做完, 直接把 status 改成 STOPPED, 并且 set shutdown_event
            self.status = InferenceEngineStatus.STOPPED
            await self.pending_queue.join()  # 等待 clean_queue() 把 queue 中的任务清理完

    def clean_queue(self):
        """
        Clean up remaining pending requests from the queue after shutdown.
        """
        while not self.pending_queue.empty():
            try:
                tokenized_data = self.pending_queue.get_nowait()
                if tokenized_data is not None:
                    self.request_store[tokenized_data.request_id].status = RequestStatus.FAILED
                    print(f"Request ID {tokenized_data.request_id} was marked as failed due to shutdown.")
                self.pending_queue.task_done()
            except asyncio.QueueEmpty:
                break

        

