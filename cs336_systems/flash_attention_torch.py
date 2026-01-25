import torch
import einops
import jaxtyping
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
    ROW, D = tensor.shape
    N_TILE = cdiv(ROW, TILE_ROW)
    tiles = torch.split(tensor, TILE_ROW, dim=0)
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
    # Get shapes
    Q_ROW, D = Q.shape
    K_ROW, D = K.shape
    V_ROW, D = V.shape
    N_TILE_q = cdiv(Q_ROW, B_q)
    N_TILE_k = cdiv(K_ROW, B_k)

    # Split the matrix
    Q_tiles = get_tiles(Q, B_q)
    K_tiles = get_tiles(K, B_k)
    V_tiles = get_tiles(V, B_k)

    # Prepare buffer
    O_final = torch.zeros((Q_ROW, D))
    L_final = torch.zeros((Q_ROW,))

    for i in range(N_TILE_q):
        # Load Q_i && Initialize statistics
        Q_i = Q_tiles[i]
        OUT_i = torch.zeros((B_q, D)) # row OUT_iput
        R_SUM_i = torch.zeros((B_q,))    # row sum
        R_MAX_i = torch.zeros((B_q,))    # row max
        # Loop over KV paris
        for j in range(N_TILE_k):
            # Compute the new tile statistics and value 
            Kt_j, V_j = rearrange(K[j], "... B_k D -> ... D B_k"), V[j]
            S_ij = einsum(Q_i, Kt_j, "... B_q D, ... D B_k -> ... B_q B_k") / D
            new_max = S_ij.max(dim=1).values    # Update row max
            P_i = exp(S_ij - new_max)   # safe softmax
            new_sum =  P_i.sum(dim=1) # Update & Correcting row sum  
            new_OUT_i = einsum(P_i, V_j, "... B_q B_k, ... B_k D -> ... B_q D")
            
            # Cache the curr iter's statistics && Update the previous iter's values
            R_SUM_i = new_sum + R_SUM_i * exp(R_MAX_i - new_max)
            R_MAX_i = new_max + 
            update_mat = torch.eye(B_q) * exp(R_MAX_i - new_max)
            OUT_i = new_OUT_i+ einsum(OUT_i, update_mat, "... B_q D, B_q B_q -> ... B_q D")

        # Recorver the original attention scores exp(Sij) without subtracting the rowmax
        OUT_i = torch.diag(1/R_SUM_i) @ OUT_i
        O_final[i*N_TILE_q:(i+1)*N_TILE_q,:] = OUT_i
        # Compute the Softmax Normalization constant
        L_i = R_MAX_i + log(R_SUM_i)
        L_final[i*N_TILE_q:(i+1)*N_TILE_q] = L_i
    
    return O, L