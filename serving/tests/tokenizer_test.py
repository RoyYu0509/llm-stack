import pytest
from transformers import AutoTokenizer
from inferenceLM.request_receiver.tokenizer import Tokenizer

@pytest.fixture(scope="module")
def tokenizer_fixture():
    return Tokenizer("gpt2")

@pytest.fixture(scope="module")
def hf_tokenizer_fixture():
    return AutoTokenizer.from_pretrained("gpt2")

def test_encode_result_match_transformers_tokenizer(tokenizer_fixture, hf_tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture
    hf_tokenizer = hf_tokenizer_fixture

    text = "Hello, world!"
    custom_tokens = custom_tokenizer.encode(text)
    hf_tokens = hf_tokenizer.encode(text)

    assert custom_tokens == hf_tokens

def test_empty_string_result_match_transformers_tokenizer(tokenizer_fixture, hf_tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture
    hf_tokenizer = hf_tokenizer_fixture

    text = ""
    custom_tokens = custom_tokenizer.encode(text)
    hf_tokens = hf_tokenizer.encode(text)

    assert custom_tokens == hf_tokens

def test_special_char_encode_result_match_transformers_tokenizer(tokenizer_fixture, hf_tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture
    hf_tokenizer = hf_tokenizer_fixture

    text = "This is a test! @#$$%^&*()"
    custom_tokens = custom_tokenizer.encode(text)
    hf_tokens = hf_tokenizer.encode(text)

    assert custom_tokens == hf_tokens   


def test_decode_result_match_transformers_tokenizer(tokenizer_fixture, hf_tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture
    hf_tokenizer = hf_tokenizer_fixture

    text = "Hello, world!"
    tokens = hf_tokenizer.encode(text)
    custom_decoded_text = custom_tokenizer.decode(tokens)
    hf_decoded_text = hf_tokenizer.decode(tokens)

    assert custom_decoded_text == hf_decoded_text


def test_decode_empty_tokens_result_match_transformers_tokenizer(tokenizer_fixture, hf_tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture
    hf_tokenizer = hf_tokenizer_fixture

    tokens = []
    custom_decoded_text = custom_tokenizer.decode(tokens)
    hf_decoded_text = hf_tokenizer.decode(tokens)

    assert custom_decoded_text == hf_decoded_text

def test_decode_special_char_tokens_result_match_transformers_tokenizer(tokenizer_fixture, hf_tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture
    hf_tokenizer = hf_tokenizer_fixture

    text = "This is a test! @#$$%^&*()"
    tokens = hf_tokenizer.encode(text)
    custom_decoded_text = custom_tokenizer.decode(tokens)
    hf_decoded_text = hf_tokenizer.decode(tokens)

    assert custom_decoded_text == hf_decoded_text


def test_encode_decode_cycle(tokenizer_fixture):
    custom_tokenizer = tokenizer_fixture

    original_text = "Testing encode-decode cycle."
    tokens = custom_tokenizer.encode(original_text)
    decoded_text = custom_tokenizer.decode(tokens)

    assert original_text == decoded_text


def test_long_text_do_not_truncate(tokenizer_fixture, hf_tokenizer_fixture):
    text = "This is a long text that should not be truncated by the tokenizer. " * 100
    encoded_tokens = tokenizer_fixture.encode(text)
    decoded_text = tokenizer_fixture.decode(encoded_tokens)
    assert text == decoded_text
