import jaxtyping

from cs336_basics.transfromer.embedding import Embedding
from cs336_basics.transfromer.linear_module import Linear
from cs336_basics.transfromer.multiheads_attention import MultiHeadsAttention
from cs336_basics.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from cs336_basics.transfromer.positionalNencoding import PosEncod
from cs336_basics.transfromer.rmsnorm import Rmsnorm
import torch.nn as nn
import torch
import torch.cuda.nvtx as nvtx

class PreNormTransformer(nn.Module):
    def __init__(self,
                 d_model: int, heads_num: int, # Multi-Head Attention args
                 dim_ff: int, # FNN args
                 pos_encod=None, token_positions=None, # Multi-Head Attention kwargs
                 eps: float = 1e-5, # rmsnorm kwargs
                 latent_exp_factor = 8/3, # FNN kwargs
                 device=None, dtype=torch.float16  # general kwargs
                 ):
        """
        Components:

            - self.RMSN: (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
                - `ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
            
            - self.MHA:  (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
                - `attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is (d_model, d_model).
            
            - self.FNN:  (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model) 
                - `ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_model, d_ff). 
        """
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.RMSN1 = Rmsnorm(d_model, eps, device, dtype)
        self.MHA = MultiHeadsAttention(d_model, heads_num, pos_encod, token_positions, device, dtype)
        self.RMSN2 = Rmsnorm(d_model, eps, device, dtype)
        self.FNN = PointwiseSGLUactFFN(d_model, dim_ff, latent_exp_factor, device, dtype)

    @nvtx.range("PreNormTransformer_forward")
    def forward(self,x:torch.Tensor, token_positions=None):
        """
        Return the output from a transformer block.
        (batch, seq, d_model) -> (batch, seq, d_model)

        Parameter:
            x: (batch, seq, d_model)
        """
        x = x + self.MHA.forward(self.RMSN1.forward(x), token_positions=token_positions)
        x = x + self.FNN.forward(self.RMSN2.forward(x))
        return x

        

        
