import pytest

from inferenceLM.engine.lm_engine import LMEngine
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from inferenceLM.data.tokenized_data import TokenizedData

@pytest.fixture(scope="module")
def model():
    return GPT2LMHeadModel.from_pretrained("gpt2")

@pytest.fixture(scope="module")
def tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, max_length",
    [
        ("Hello, how are you? My name is Yifan and ", 30),    # short_seq
        ("Hello, how are you? My name is Yifan and ", 1024),  # long_seq
        ("Yifan Yu is ", 200),                                # random_prompt
    ],
    ids=["short_seq", "long_seq", "random_prompt"]
)
async def test_lm_engine_matches_huggingface(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, max_length: int):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    hf_output = model.generate(  # type: ignore
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        max_length=max_length,
    )
    hf_new_tokens = hf_output[0].tolist()[input_ids.shape[1]:]

    tokenized_data = TokenizedData(request_id="test_request", tokens=input_ids[0].tolist())
    engine_new_tokens = await LMEngine(model).inference(tokenized_data, do_sample=False, max_length=max_length)

    assert hf_new_tokens == engine_new_tokens

@pytest.mark.asyncio
async def test_lm_engine_reject_invalid_max_len(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    max_length = 1025
    do_sample = False
    user_input = "Hello, how are you? My name is Yifan and "
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids

    tokenized_data = TokenizedData(request_id="test_request", tokens=input_ids[0].tolist())
    lm_engine = LMEngine(model)
    with pytest.raises(RuntimeError, match=f"User defined max_length {max_length} exceeds the model's official maximum context length {model.config.n_positions}."):
        await lm_engine.inference(tokenized_data, do_sample=do_sample, max_length=max_length)

@pytest.mark.asyncio
async def test_lm_engine_prompt_equals_max_len_prefill_should_reject(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    # Load from huggingface
    max_length = 10
    do_sample = False
    user_input = "Hello, how are you? My name is Yifan and "
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids[:, :max_length] # truncate the input to max_length
    
    # must be exactly max_length
    assert input_ids.shape[1] == max_length, f"Input token sequence length {input_ids.shape[1]} should equal max_length {max_length}"

    tokenized_data = TokenizedData(request_id="test_request", tokens=input_ids[0].tolist())
    lm_engine = LMEngine(model)
    with pytest.raises(RuntimeError, match=f"Input token sequence length {input_ids.shape[1]} equals max_length {max_length}, no space for generation."):
        await lm_engine.inference(tokenized_data, do_sample=do_sample, max_length=max_length)
