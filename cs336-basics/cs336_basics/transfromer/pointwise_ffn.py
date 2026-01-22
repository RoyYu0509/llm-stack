from cs336_basics.transfromer.para_init import trunct_normal_para_init
import torch
from jaxtyping import Float
from einops import einsum
import torch.cuda.nvtx as nvtx

class PointwiseSGLUactFFN(torch.nn.Module):
    def __init__(self, dim_model, dim_ff=None, latent_exp_factor = 8/3 , device="cpu", dtype=torch.float16):
        super().__init__()
        # Round down to the nearest dimension of the multiple of 64
        if dim_ff is None:
            dim_ff = (latent_exp_factor * dim_model // 64) * 64
        self.W1: Float[torch.Tensor, "dim_ff, d_model"]= torch.nn.Parameter(trunct_normal_para_init(dim_ff, dim_model, device=device, dtype=dtype))
        self.W3: Float[torch.Tensor, "dim_ff, d_model"]= torch.nn.Parameter(trunct_normal_para_init(dim_ff, dim_model, device=device, dtype=dtype))
        self.W2: Float[torch.Tensor, "d_model, dim_ff"]= torch.nn.Parameter(trunct_normal_para_init(dim_model, dim_ff, device=device, dtype=dtype))

    def SiLU(self, x: torch.Tensor):
        """
        SiLU Activation Function:  x/(1+torch.exp(-x))
        
        Parameter:
            x: A vector of (batch, seq, d_model)
        """
        
        return x * torch.sigmoid(x)
    
    @nvtx.range("PointwiseSGLUactFFN_forward")
    def forward(self, x: torch.Tensor):
        """
        One FF pass.
        """
        x_copy = x.clone()
        # W1 x
        x = einsum(self.W1, x, "dim_ff d_model, ... d_model -> ... dim_ff")
        # SiLU(W1x)
        x = self.SiLU(x)
        # Gating: SiLU(W1x)⊙W3x 
        x = x * einsum(self.W3, x_copy, "dim_ff d_model, ... d_model -> ... dim_ff")
        # Output: Project back
        return einsum(self.W2, x, "d_model dim_ff, ... dim_ff -> ... d_model")
    

