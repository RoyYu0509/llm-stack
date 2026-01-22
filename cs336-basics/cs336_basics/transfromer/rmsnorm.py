import torch
from einops import reduce, einsum
from jaxtyping import Float, Int
from einops import rearrange
import torch.cuda.nvtx as nvtx


class Rmsnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. 

        Parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype 

        self.eps = eps       

        gain = torch.ones(d_model, dtype=self.dtype, device=self.device)
        self.gain: Float[torch.Tensor, "d_model"] = torch.nn.Parameter(gain)
    
    @nvtx.range("Rmsnorm_forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        
        Return a tensor of the same shape.

        FLOPs:
            - Square: (batch_size * sequence_length * d_model)
            - Mean: (batch_size * sequence_length) * (d_model+1)
            - Add eps: (batch_size * sequence_length * d_model) 
            - Compute rms_x
        """
        # Up-scale percision
        x_dtype = x.dtype
        x: Float[torch.tensor, "batch_size, sequence_length, d_model"]= x.to(torch.float32)

        # Compute Mean Squared
        squared_x_gain: Float[torch.Tensor, "batch_size, sequence_length, d_model"] = x.pow(2)
        
        # Compute RMS
        mean_squared_sum_x = reduce(squared_x_gain, "... d_model -> ...", "mean")
        # Keep eps on the same device as the activation to avoid device mismatches (e.g., mps vs mps:0).
        eps = torch.as_tensor(self.eps, device=mean_squared_sum_x.device, dtype=torch.float32)
        rms_x: Float[torch.Tensor, "batch_size, sequence_length, 1"]= torch.sqrt(mean_squared_sum_x + eps).unsqueeze(-1)
        
        return ((x)/rms_x * self.gain).to(x_dtype) # scale back to input dtype
