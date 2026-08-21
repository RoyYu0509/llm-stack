import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int

def trunct_normal_para_init(dim_out, dim_in, mu=0, var=1, truc_magnitude=3, device="cpu", dtype=torch.float16):
    """
    Return the initialized parameters value using truncated normal.
    """
    device = device if device is not None else "cpu"
    dtype = dtype if dtype is not None else torch.float16 
    tensor = torch.empty(dim_out, dim_in, dtype = dtype, device = device)
    weight_mat: Float[torch.Tensor, "dim_out, dim_in"] = tensor 
    nn.init.trunc_normal_(
        weight_mat, mean=mu, std=math.sqrt(var), 
        a=mu-truc_magnitude, b=mu+truc_magnitude,
    )

    return weight_mat
    