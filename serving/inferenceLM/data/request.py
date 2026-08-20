from inferenceLM.data.request_status import RequestStatus
from pydantic import BaseModel, Field
from typing import List

class RequestData(BaseModel):
    """
    A class to represent the data associated with a message.

    Attributes:
        user_id (str): The ID of the user who sent the message.
        request_id (str): The unique ID of the message.
        timestamp (float): The time when the message was created.
        status (RequestStatus): The current status of the message.
        prompt_text (str): The raw text prompts associated with the message.
    
    Methods:
        __str__(): Returns a string representation of the RequestData object.
        serialization(): Serializes the RequestData object to JSON format.
        deserialization(json_str): Deserializes a JSON string back into a RequestData object.
    """
    user_id: str
    request_id: str
    timestamp: float
    status: RequestStatus = RequestStatus.PENDING
    prompt_text: str
    max_token_length: int = 20  # prompt + response <= max_token_length
    do_sample: bool = False
    generated_tokens: List[int] = Field(default_factory=list)  # 提前准备buffer
    
    def __str__(self):
        return f"RequestData(\nuser_id={self.user_id}, \nrequest_id={self.request_id}, \ntimestamp={self.timestamp}, \nstatus={self.status}, \nprompt_text={self.prompt_text}, \nmax_token_length={self.max_token_length}, \ndo_sample={self.do_sample} \n)"

    def serialization(self):
        return self.model_dump_json()

    @classmethod
    def deserialization(cls, json_str: str):
        return cls.model_validate_json(json_str)
        