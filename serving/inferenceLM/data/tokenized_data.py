class TokenizedData:
    """
    A class to represent the tokenized data of a request.

    Attributes:
        request_id (str): The unique ID of the request.
        tokens (List[int]): The tokenized input data for the request.

    Methods:
        __str__(): Returns a string representation of the TokenizedData object.
    """
    def __init__(self, request_id: str, tokens: list[int]):
        self.request_id = request_id
        self.tokens = tokens # the tokenized input data for the request

    def __str__(self):
        return f"Tokenized request: {self.request_id}"

    def __len__(self):
        return len(self.tokens)
    

