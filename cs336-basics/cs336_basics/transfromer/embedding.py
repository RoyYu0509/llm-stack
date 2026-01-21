from cs336_basics.transfromer.para_init import trunct_normal_para_init
import torch
from jaxtyping import Float, Int
from einops import rearrange, einsum
import torch.cuda.nvtx as nvtx

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        An embedding module.

        Embeding a token is to map a scalar token_ID -> to a d dimensional vector.
        The correspondent relationship is stored as an embedding matrix attribute.
        i-th row corresponds to the vector representation of the i-th vocabulary.
        
        Attributes:
           embed_mat:       Float[torch.Tensor, "vocab_size embedding_dim"]
        
        Parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim:  int Dimension of the embedding vectors, i.e., dmodel
            device:         torch.device | None = None Device to store the parameters on
            dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        embed_mat = trunct_normal_para_init(num_embeddings, embedding_dim,0, 1, 3, device, dtype)
        self.embed_mat: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.nn.Parameter(embed_mat)
        self.device = device
        self.dtype = dtype

    @nvtx.range("Embedding_forward")
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Return the embedding vectors of the given token IDs.

        Loop up in the Word Embedding Matrix
        """
        # Simple row retrieving
        return self.embed_mat[token_ids]


    