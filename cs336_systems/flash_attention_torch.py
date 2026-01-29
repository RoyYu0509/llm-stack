import torch
import einops
import jaxtyping
from jaxtyping import Float
from torch import Tensor
import math
from math import ceil as cdiv
from einops import rearrange, einsum

def get_tiles(tensor, TILE_ROW):
    """
    Returns a list of sub-tensor blocks, idx = [0, ..., n_TILE].
    Each sub-tensor is of shape = [TILE_ROW, TENSOR_D], with padding
    for OUT_i of bound array section.

    Each tile is of size [TILE_ROW, D], except for the last one shape = [r, D]
    where r = ROW % TILE_ROW is the remainder rows.
    """
    ROW, D = tensor.shape[-2:]
    N_TILE = cdiv(ROW/TILE_ROW)
    # Split on the sequence dimension
    tiles = torch.split(tensor, TILE_ROW, dim=-2)
    return tiles


def flash_attention_torch_fwd(
    Q, K, V,
    B_q, B_k
):
    """
    Return the OUT_iput of the attention operation ATTEN(QK.T) V

    Parameters:
        - Q: Float[torch.Tensor, "N_q, d"] The Query matrix
        - K: Float[torch.Tensor, "N_k, d"] The Key matrix
        - V: Float[torch.Tensor, "N_k, d"] The Value matrix
        - B_q: int Query TILE_ROW
        - B_k: int Key TILE_ROW
    """
    device = Q.device
    dtype = Q.dtype
    
    # Get shapes
    BATCH_SHAPE = Q.shape[:-2]
    Q_ROW, D = Q.shape[-2:]
    K_ROW, D = K.shape[-2:]
    V_ROW, D = V.shape[-2:]
    N_TILE_q = cdiv(Q_ROW / B_q)
    N_TILE_k = cdiv(K_ROW / B_k)

    # Split the matrix
    Q_tiles = get_tiles(Q, B_q)
    K_tiles = get_tiles(K, B_k)
    V_tiles = get_tiles(V, B_k)

    # Prepare buffer on correct device
    O_final = torch.zeros((*BATCH_SHAPE, Q_ROW, D), device=device, dtype=dtype)
    L_final = torch.zeros((*BATCH_SHAPE, Q_ROW, 1), device=device, dtype=dtype)

    for i in range(N_TILE_q):
        # Load Q_i && Initialize statistics
        Q_i     :Float[Tensor, "... B_q, D"]    = Q_tiles[i]
        OUT_i   :Float[Tensor, "... B_q, D"]    = torch.zeros((*BATCH_SHAPE, B_q, D), device=device, dtype=dtype)  # row OUT_iput
        R_SUM_i :Float[Tensor, "... B_q, 1"]    = torch.zeros((*BATCH_SHAPE, B_q, 1), device=device, dtype=dtype)    # row sum
        R_MAX_i :Float[Tensor, "... B_q, 1"]    = torch.zeros((*BATCH_SHAPE, B_q, 1), device=device, dtype=dtype)    # row max
        # Loop over KV paris
        for j in range(N_TILE_k):
            # print("Go to tile:", j)
            V_j: Float[Tensor, "... B_k, D"] = V_tiles[j]
            K_j: Float[Tensor, "... B_k, D"] = K_tiles[j]
            Kt_j:Float[Tensor, "... D, B_k"] = rearrange(K_j, "... B_k D -> ... D B_k")

            # Compute the new tile statistics and value 
            S_ij:     Float[Tensor, "... B_q, B_k"]     = einsum(Q_i, Kt_j, "... B_q D, ... D B_k -> ... B_q B_k") / (D**0.5)
            new_max:  Float[Tensor, "... B_q, 1"]       = S_ij.max(dim=-1, keepdim=True).values   # Update row max
            P_i:      Float[Tensor, "... B_q, B_k"]     = torch.exp(S_ij - new_max)  # safe softmax
            new_sum:  Float[Tensor, "... B_q, 1"]       =  P_i.sum(dim=-1, keepdim=True) # Update & Correcting row sum  
            new_OUT_i:Flaot[Tensor, "... B_q, D"]       = einsum(P_i, V_j, "... B_q B_k, ... B_k D -> ... B_q D") # Compute the new output tile
            
            # Cache the curr iter's statistics && Update the previous iter's values
            # print("new_sum:", new_sum.shape)
            # print("torch.exp(R_MAX_i - new_max):", torch.exp(R_MAX_i - new_max).shape)
            # print("new_max", new_max.shape)
            # print("R_SUM_i:", R_SUM_i.shape)
            R_SUM_i:        Float[Tensor, "... B_q, 1"] = new_sum + R_SUM_i * torch.exp(R_MAX_i - new_max)
            correct_sacle:  Float[Tensor, "... B_q, 1"] = torch.exp(R_MAX_i - new_max)
            OUT_i:          Float[Tensor, "... B_q, D"] = new_OUT_i + OUT_i * correct_sacle
            R_MAX_i:        Float[Tensor, "... B_q, 1"] = new_max

        # Perform Softmax Normalization
        OUT_i = OUT_i / R_SUM_i
        start = i*B_q
        end = (i+1)*B_q
        O_final[..., start:end, :] = OUT_i
        # Compute LogSumExp for later backward pass
        L_i = R_MAX_i + torch.log(R_SUM_i)
        L_final[..., start:end, :] = L_i

    return O_final, L_final.squeeze(-1)

    

class FlashAttentionTorchFunctionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass for FlashAttention using PyTorch operations.

        Parameters:
            - Q: Float[torch.Tensor, "N_q, d"] The Query matrix
            - K: Float[torch.Tensor, "N_k, d"] The Key matrix
            - V: Float[torch.Tensor, "N_k, d"] The Value matrix
            - B_q: int Query TILE_ROW
            - B_k: int Key TILE_ROW
        """
        B_q, B_k= 16, 16  # Example tile sizes; these can be tuned based on hardware
        O, L = flash_attention_torch_fwd(Q, K, V, B_q, B_k)
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.is_causal = is_causal
        ctx.B_q = B_q
        ctx.B_k = B_k
        return O

    @staticmethod
    def backward(ctx, dLdO):
        """
        Backward pass for FlashAttention implemented with PyTorch. (套公式，没啥看的)

        Parameters:
            - dLdO: Gradient of the output from the forward pass.
        """
        Q, K, V, O, L = ctx.saved_tensors
        L: Float[Tensor, "B Q_N"]
        is_causal = ctx.is_causal
        D:      Float[Tensor, "B, Q_N, 1"] = torch.sum(O*dLdO, dim=-1, keepdim=True) 
        d = Q.shape[-1]

        # Compute S, P with L (log-sum-exp)
        KT = rearrange(K, "B N D -> B D N")
        S:      Float[Tensor, "B Q_N N_K"] = Q@KT / d**0.5
        P:      Float[Tensor, "B Q_N N_K"] = torch.exp(S-L[:,:,None])
        
        # Apply causal mask if needed
        if is_causal:
            B, Q_N, K_N = S.shape
            mask = torch.tril(torch.ones(Q_N, K_N, device=Q.device, dtype=torch.bool))
            P = P * mask[None, :, :]
        
        # Compute gradients
        PT:     Float[Tensor, "B K_N Q_N"] = rearrange(P, "B Q_N K_N -> B K_N Q_N")
        dLdV:   Float[Tensor, "B K_N D"] = PT@dLdO
        
        VT:     Float[Tensor, "B D K_N"] = rearrange(V, "B K_N D -> B D K_N")
        dLdP:   Float[Tensor, "B Q_N N_K"] = dLdO@VT
        
        dLdS:   Float[Tensor, "B Q_N N_K"] = P * (dLdP - D)
        dLdQ:   Float[Tensor, "B Q_N D"] = dLdS@K / d**0.5
        dLdST:  Float[Tensor, "B N_K Q_N"] = rearrange(dLdS, "B Q_N N_K -> B N_K Q_N")
        dLdK:   Float[Tensor, "B N_K D"] = dLdST @ Q / d**0.5

        return dLdQ, dLdK, dLdV, None
    

flash_attn_torch_fn = FlashAttentionTorchFunctionTorch.apply