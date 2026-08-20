import inferenceLM.data.tokenized_data as tokenized_data

def test_tokenized_data_str():
    tokenized = tokenized_data.TokenizedData(
        request_id="test_request",
        tokens=[0, 1],
    )
    expected = f"Tokenized request: {tokenized.request_id}"
    assert str(tokenized) == expected

def test_empty_tokens():
    tokenized = tokenized_data.TokenizedData(
        request_id="test_request",
        tokens=[],
    )
    assert len(tokenized) == 0

def test_non_empty_tokens():
    tokenized = tokenized_data.TokenizedData(
        request_id="test_request",
        tokens=[0, 1, 2],
    )
    assert len(tokenized) == 3


