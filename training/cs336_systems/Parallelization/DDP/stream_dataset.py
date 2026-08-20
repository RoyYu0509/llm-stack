import numpy as np
import torch
from torch.utils.data import Dataset

class TokenStreamDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads a stream of tokens from a numpy .npy file.
    
    Designed for DDP LM training Task.

    Parameters:
        - path: file path to the token stream (numpy .npy format)
        - context_len: the context length per token sequence sample, ie. tok_strm[i:i+context_len].
    """
    def __init__(self, path: str, context_len: int):
        self.tokens_stream = np.load(path, mmap_mode="r")
        self.context_len = context_len
        self.N = self.tokens_stream.shape[0]

        if self.N <= self.context_len:
            raise RuntimeError("context_length is too large for the provided data.")

    def __len__(self) -> int:
        return self.N - self.context_len - 1

    def __getitem__(self, idx: int):
        x_np = self.tokens_stream[idx : idx + self.context_len]
        y_np = self.tokens_stream[idx + 1 : idx + 1 + self.context_len]

        # Convert only this sample to int64 for embedding lookup.
        x = torch.tensor(x_np, dtype=torch.long)
        y = torch.tensor(y_np, dtype=torch.long)
        return x, y