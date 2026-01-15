from cs336_basics.transfromer.para_init import trunct_normal_para_init
import torch
from jaxtyping import Float, Int
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, dim_feats, dim_out, device="cpu", dtype=torch.float16):
        """
        linear transformation module. This function should accept the following parameters:
        
        Parameters:
            in_features:    int final dimension of the input
            out_features:   int final dimension of the output
            device:         torch.device | None = None Device to store the parameters on
            dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        weightMat = trunct_normal_para_init(dim_out, dim_feats, 
                                            0, 2/(dim_out+dim_feats), 3, 
                                            device, dtype)
        self.weightMat: Float[torch.Tensor, "dim_out, dim_feats"] = torch.nn.Parameter(weightMat)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        # Convert x datatype
        x = x.to(self.dtype)
        return einsum(self.weightMat, x, "dim_out dim_feats, ... dim_feats -> ... dim_out")


