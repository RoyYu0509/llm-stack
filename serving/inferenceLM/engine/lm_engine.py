import torch, asyncio
from transformers import AutoTokenizer, PreTrainedModel
from jaxtyping import Float
from typing import Tuple, List, Dict
from inferenceLM.data.request import RequestData
from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData

class LMEngine:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.model.eval()  # 直接开启 eval mode

    def prefill(
            self, 
            input_ids: torch.Tensor, 
            do_sample: bool = False, 
            max_length: int = 20
        ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Prefill step: 输入一个 token sequence, 进行一次 forward pass, 生成 new_token 和 new_kv.

        Args:
            input_ids (torch.Tensor): 输入的 token sequence, shape (1, seq_len)
            do_sample (bool): 是否使用采样策略
            max_length (int): prompt tokens 加 generated tokens 的总长度

        Returns:
            new_token (torch.Tensor): 生成的 new_token, shape (1,)
            new_kv (List[Tuple[torch.Tensor, torch.Tensor]]): 生成的 new_kv, list of tuples with shape (1, seq_len, d_model)
        """
        if input_ids.shape[1] >= max_length:
            raise RuntimeError(f"Input token sequence length {input_ids.shape[1]} exceeds max_length {max_length}.")

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, past_key_values=None)
            new_logits: Float[torch.Tensor, "1, Vocab"] 
            new_logits = outputs.logits[:, -1, :]  
            new_kv: List[Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]] 
            new_kv = outputs.past_key_values
        
        if not do_sample:
            new_token = torch.argmax(new_logits, dim=-1).unsqueeze(1) # shape (1, 1)
        else:
            raise NotImplementedError(f"Sampling strategy {do_sample} not implemented.")
        
        return new_token, new_kv


    def decode(
            self, 
            latest_token: torch.Tensor, 
            latest_kv: List[Tuple[torch.Tensor, torch.Tensor]], 
            do_sample: bool = False, 
        ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Decode step: 输入 latest_token 和 latest_kv, 生成 new_token.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=latest_token, past_key_values=latest_kv, use_cache=True)
            new_logits: Float[torch.Tensor, "1, Vocab"] 
            new_logits = outputs.logits[:, -1, :]  
            new_kv: List[Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]] 
            new_kv = outputs.past_key_values

        if not do_sample:
            new_token = torch.argmax(new_logits, dim=-1).unsqueeze(1) # shape (1, 1)
            return new_token, new_kv
        else:
            raise NotImplementedError(f"Sampling strategy {do_sample} not implemented.")
        

    async def inference(self, tokenized_data: TokenizedData, do_sample: bool = False, max_length: int = 20) -> List[int]:
        """
        Return the generated token sequence as output, given the tokenized input data.
        the return token sequence do not include prompt tokens, only the generated tokens.

        prompt_length + generated_length <= max_length

        Args:
            tokenized_data (TokenizedData): The tokenized data for the request, containing input tokens
            do_sample (bool): Whether to use sampling strategy
            max_length (int): The maximum length of the generated sequence

        Returns:
            generated_tokens (List[int]): The generated token sequence as output.
        """
        # Check if the max_len is gerater than model's official max_len
        model_capacity = self.model.config.n_positions
        if max_length > model_capacity:
            raise RuntimeError(f"User defined max_length {max_length} exceeds the model's official maximum context length {model_capacity}.")

        # Generated Tokens' buffer
        generated_tokens = []
        token_tensor = torch.tensor(tokenized_data.tokens).unsqueeze(0)  # shape (1, seq_len)

        # Prefill step
        if token_tensor.shape[1] == max_length:
            raise RuntimeError(f"Input token sequence length {token_tensor.shape[1]} equals max_length {max_length}, no space for generation.")
        latest_token, latest_kv = self.prefill(token_tensor, do_sample=do_sample, max_length=max_length)
        await asyncio.sleep(0)  # 先做 0 秒钟的别的事, 等一下 prefill
        generated_tokens.append(latest_token.item())

        # Decode step (用一个 cls.method() 来检查是不是 halt 生成)
        while not self.stopping_criteria(len(tokenized_data), generated_tokens, max_length):
            latest_token, latest_kv = self.decode(latest_token, latest_kv, do_sample=do_sample)
            await asyncio.sleep(0)  # 先做 0 秒钟的别的事, 等一下 decode
            generated_tokens.append(latest_token.item())

        return generated_tokens
    

    def stopping_criteria(self, prompt_len: int, generated_tokens: List[int], max_length: int) -> bool:
        """
        Return True (means stop) if:
            1. generated_tokens reaches max length (e.g. 20)
            2. the last generated token is the EOS token.
        """
        return len(generated_tokens) + prompt_len >= max_length or generated_tokens[-1] == self.model.config.eos_token_id
