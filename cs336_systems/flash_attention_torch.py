import torch
import einops
import jaxtyping
import math
from math import ceil as cdiv

def get_tiles(tensor, TILE_ROW):
    """
    Returns a list of sub-tensor blocks, idx = [0, ..., n_TILE].
    Each sub-tensor is of shape = [TILE_ROW, TENSOR_D], with padding
    for out of bound array section.
    """
    N_TILE = cdiv()
    for i in range():


def flash_attention_torch_fwd(
    Q, K, V,
    B_q, B_k
):
    """
    Return the output of the attention operation ATTEN(QK.T) V

    Parameters:
        - Q: Float[torch.Tensor, "N_q, d"] The Query matrix
        - K: Float[torch.Tensor, "N_k, d"] The Key matrix
        - V: Float[torch.Tensor, "N_k, d"] The Value matrix
        - B_q: int Query TILE_ROW
        - B_k: int Key TILE_ROW
    """
    # Get shapes
    Q_ROW, Q_D = Q.shape
    K_ROW, K_D = K.shape
    V_ROW, V_D = V.shape
    N_TILE_q = cdiv(Q_ROW, B_q)
    N_TILE_k = cdiv(K_ROW, B_k)

    # 