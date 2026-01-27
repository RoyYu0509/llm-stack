"""
- backward pass
- connect to pytorchgraph
- re-use intermediate values from fwd & bwd pass
- locks & atomics operations
- two sequential kernels (Better Than) one fused kernel
"""

import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def layer_norm_triton():
    pass




# Testing against torch implementation
def test_layernorm_kernel(
    X_Shape:tuple, # 2D input
    dtype,
    eps=1e-5,
    device=DEVICE      
):  
    N, D = X_Shape
    X = -2.3 + 0.5 * torch.randn(X_Shape, dtype=dtype, device=DEVICE)
    X.requires_grad_(True)
    w = torch.rand((N,), dtype=dtype, device=DEVICE)
    b = torch.rand((N,), dtype=dtype, device=DEVICE)

    print("Testing Forward Prop")
    y_triton = layer_norm_triton(X, (N, ), w, b, eps)
    y_torch = torch.nn.functional.layer_norm(X, (N,), w, b, eps).to(dtype)
    torch.testing.assert_close(y_triton, y_torch)
    print("Forward pass test PASS!")
    

    print("Testing Back Prop")
    dLdy = 0.1 * torch.randn_like(X)
    y_triton.backward(dLdy, retain_graph = True) # Retain the graph, or the PyTorch will reset the graph after call backprop
    dLdx_tri, dLdw_tri, dLdb_tri = [tensor.grad.clone() for tensor in [X, w, b]]
    # Reset the gradient
    X.grad, w.grad, b.grad = None, None, None,
    y_torch.backward(dLdy, retain_graph=True)
    dLdx_tor, dLdw_tor, dLdb_tor = [tensor.grad.clone() for tensor in [X, w, b]]

    torch.testing.assert_close(dLdx_tor, dLdx_tri, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_tor, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_tor, atol=1e-2, rtol=0)

    print("Backward pass test PASS!")

