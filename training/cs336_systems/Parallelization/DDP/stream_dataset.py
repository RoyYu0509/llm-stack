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
        - stride: distance between the start offsets of consecutive samples.
          1 (default) = every offset is a sample (overlapping sliding windows, the old behaviour).
          context_len = non-overlapping chunks: one epoch is exactly one pass over every token,
          so "how many times did the model see each token" is a number you can state. It also
          keeps len() ~N/context_len: DistributedSampler materialises randperm(len).tolist() on
          every rank, which at stride 1 is ~36 bytes of Python ints per TOKEN (~29GB of host RAM
          per rank for an 800M-token corpus).
    """
    def __init__(self, path: str, context_len: int, stride: int = 1):
        self.tokens_stream = np.load(path, mmap_mode="r")
        self.context_len = context_len
        self.stride = stride
        self.N = self.tokens_stream.shape[0]

        if stride < 1:
            raise ValueError("stride must be >= 1")
        if self.N <= self.context_len:
            raise RuntimeError("context_length is too large for the provided data.")

    def __len__(self) -> int:
        if self.stride == 1:
            return self.N - self.context_len - 1
        # starts 0, stride, 2*stride, ... with start + context_len + 1 <= N (y needs one extra token)
        return (self.N - self.context_len - 1) // self.stride + 1

    def __getitem__(self, idx: int):
        idx = idx * self.stride
        x_np = self.tokens_stream[idx : idx + self.context_len]
        y_np = self.tokens_stream[idx + 1 : idx + 1 + self.context_len]

        # Convert only this sample to int64 for embedding lookup.
        x = torch.tensor(x_np, dtype=torch.long)
        y = torch.tensor(y_np, dtype=torch.long)
        return x, y