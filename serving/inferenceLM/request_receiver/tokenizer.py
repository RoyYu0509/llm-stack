from transformers import AutoTokenizer

class Tokenizer:
    """
    Tokenizer class for tokenizing input text for language model inference, backed by 
    Hugging Face's transformers library.

    Attributes:
        tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    """
    def __init__(self, tokenizer_name: str):
        """
        Initialize the Tokenizer with a specified Hugging Face tokenizer.

        Args:
            tokenizer_name (str): The name of the Hugging Face tokenizer to use (e.g. "gpt2", "bert-base-uncased").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, text: str) -> list[int]:
        """
        Encode input text using the Hugging Face tokenizer.
        """
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back into text using the Hugging Face tokenizer.
        """
        return self.tokenizer.decode(token_ids)

