import torch
from jaxtyping import Float, Array
from torch import Tensor
from einops import rearrange, reduce, repeat, einsum
from cs336_basics.transfromer.para_init import trunct_normal_para_init
from cs336_basics.transfromer.scaled_dot_prod_attention import softmax, scaled_dot_product_attention
import torch.cuda.nvtx as nvtx


class MultiHeadsAttention(torch.nn.Module):
    def __init__(self,d_model: int, heads_num: int, pos_encod=None, token_positions=None, device=None, dtype=torch.float16):
        """
        A Multi-Heads Attention Module

        Parameters:
            d_model:    int Dimensionality of input token vector.
            num_heads:  int Number of heads to use in multi-head self-attention.
            pos_encod:  Encoder A Encoder object
            token_positions: torch.Tensor A tensor specifying the position of the token vecor for z

        Attributes:
            heads_num:  int Number of Attention Heads
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads_num = heads_num
        self.d_model = d_model
        self.d_k = d_model//heads_num
        self.d_v = d_model//heads_num 

        # Define the projection matrix
        W_dim_in = int(heads_num*self.d_k)
        W_dim_out = d_model
        W_Q: Float[Tensor, "(h d_k) d_model"] = trunct_normal_para_init(
            W_dim_in, W_dim_out, 
            mu=0, var=2/(W_dim_out+W_dim_in), truc_magnitude=3,
            device=device, dtype=dtype
        )
        self.W_Q = torch.nn.Parameter(W_Q)
        # self.W_Q = torch.nn.Parameter(rearrange(self.W_Q, "(h d_k) d_model -> h d_k d_model", 
        #                               h = self.heads_num, d_k = self.d_k))

        W_K: Float[Tensor, "(h d_k) d_model"] = trunct_normal_para_init(
            W_dim_in, W_dim_out, 
            mu=0, var=2/(W_dim_out+W_dim_in), truc_magnitude=3,
            device=device, dtype=dtype
        )
        self.W_K = torch.nn.Parameter(W_K)
        # self.W_K = torch.nn.Parameter(rearrange(self.W_K, "(h d_k) d_model -> h d_k d_model",
        #                               h = self.heads_num, d_k = self.d_k))

        W_V: Float[Tensor, "(h d_v) d_model"] = trunct_normal_para_init(
            W_dim_in, W_dim_out, 
            mu=0, var=2/(W_dim_out+W_dim_in), truc_magnitude=3,
            device=device, dtype=dtype
        )
        self.W_V = torch.nn.Parameter(W_V) 
        # self.W_V = torch.nn.Parameter(rearrange(self.W_V, "(h d_v) d_model -> h d_v d_model",
        #                               h = self.heads_num, d_v = self.d_v))

        # Define the Out Matrix
        Out_dim_in = d_model
        Out_dim_out = int(heads_num*self.d_v)
        W_O: Float[Tensor, "d_model (h d_v)"] = trunct_normal_para_init(
            Out_dim_in, Out_dim_out, 
            mu=0, var=2/(W_dim_out+W_dim_in), truc_magnitude=3,
            device=device, dtype=dtype
        )
        self.W_O = torch.nn.Parameter(W_O)  

        # Add Positional Encoding
        self.pos_encod = pos_encod 
        self.token_positions = token_positions  # optional fallback

    @nvtx.range("MultiHeadsAttention computation")
    def _multiHead(self, x, token_positions=None):
        """
        Return the result of MultiHead(W_Q, W_K, W_V, x) 

        Return:
            - multi_head: Float[Tensor, f"... h seq d_v"]
        """
        x: Float[Tensor, f"... seq d_model"]  
        
        Q: Float[Tensor, f"... seq h d_k d_model"]
        W_Q = rearrange(self.W_Q, "(h d_k) d_model -> h d_k d_model", h = self.heads_num, d_k = self.d_k)
        Q = einsum(W_Q, x, "h d_k d_model, ... seq d_model -> ... h seq d_k")
        
        K: Float[Tensor, f"... seq h d_k d_model"]
        W_K = rearrange(self.W_K, "(h d_k) d_model -> h d_k d_model", h = self.heads_num, d_k = self.d_k)
        K = einsum(W_K, x, "h d_k d_model, ... seq d_model -> ... h seq d_k")

        V: Float[Tensor, f"... seq h d_v d_model"]
        W_V = rearrange(self.W_V, "(h d_v) d_model -> h d_v d_model", h = self.heads_num, d_v = self.d_v)
        V = einsum(W_V, x, "h d_v d_model, ... seq d_model -> ... h seq d_v")

        # Add position encoding
        positions = token_positions if token_positions is not None else self.token_positions
        if self.pos_encod is not None and positions is not None:
            Q = self.pos_encod.forward(Q, positions)
            K = self.pos_encod.forward(K, positions)

        bool_mask = self._build_mask(x)

        multi_head: Float[Tensor, f"... h seq d_v"]
        multi_head = scaled_dot_product_attention(Q, K, V, bool_mask)
        
        return multi_head
    
    def _build_mask(self, x):
        """
        Return a bool mask for the batch x, True -> Attend To.

        Parameters:
            - x: Float[Tensor, "... seq d_model"]

        Return:
            - mask: Bool[Tensor, "... seq, seq"]
        """
        seq_len = x.shape[-2]
        return ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device=x.device)  
    
    @nvtx.range("MultiHeadsAttention_forward")
    def forward(self, x, token_positions=None):
        """
        Return tensor: W_O @ MultiHead(W_Q, W_K, W_V, x) 
        """
        multi_head_attention: Float[Tensor, "... h seq_q d_v"]
        multi_head_attention = self._multiHead(x, token_positions)

        multi_head_attention = rearrange(multi_head_attention, "... h seq d_v -> ... seq (h d_v)")
        self.W_O: Float[Tensor, "d_model (h_d_v)"]
        with nvtx.range("Output_projection"):
            out = einsum(self.W_O, multi_head_attention, "d_model (h_d_v), ... seq (h_d_v) -> ... seq d_model")
        return out
