import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, reduce, repeat, einsum
from cs336_basics.transfromer.para_init import trunct_normal_para_init
from cs336_basics.transfromer.scaled_dot_prod_attention import softmax, scaled_dot_product_attention, flash_attention_my_triton, vectorized_attention_torch
from cs336_basics import nvtx_compat as nvtx


class MultiHeadsAttention(torch.nn.Module):
    def __init__(
        self,d_model: int, heads_num: int, 
        pos_encod=None, token_positions=None, 
        is_causal: bool = True,
        attention_fn=scaled_dot_product_attention, 
        device=None, dtype=torch.float16
        ):
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
        self.attention_fn = attention_fn
        self.is_causal = is_causal

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
            Q = self.pos_encod(Q, positions)
            K = self.pos_encod(K, positions)

        # Reshape to 3D by combining Batch & Head Dimensions: (B, H, seq, d_k) -> (B*H, seq, d_k)
        Q_packed = rearrange(Q, "... h seq d_k -> (... h) seq d_k")
        K_packed = rearrange(K, "... h seq d_k -> (... h) seq d_k")
        V_packed = rearrange(V, "... h seq d_v -> (... h) seq d_v")

        # Ship the compressed QKV (shape = [B*H, seq, d_k]) to the attention kernel
        multi_head_packed: Float[Tensor, f"(... h) seq d_v"]
        multi_head_packed = self.attention_fn(Q_packed, K_packed, V_packed, is_causal=self.is_causal)
        
        # Reshape back to (B, H, Seq, d_v)
        multi_head: Float[Tensor, f"... h seq d_v"]
        multi_head = rearrange(multi_head_packed, "(b h) seq d_v -> b h seq d_v", h=self.heads_num)
        
        return multi_head

    
    def forward_with_cache(self, x, token_positions, past_kv=None, attn_mask=None):
        """Incremental attention for autoregressive decoding. Inference only.

        Deliberately a separate method rather than a branch inside `forward`: the training
        path is the hot path and stays byte-identical.

        现在 forward 方法也接 token position 来同时处理多条 decode request, 但是每条 request 的 token position 都不一样, 所以需要在 forward_with_cache 里传入 token_positions 来让每条 request 都能拿到各自的 position, 以便 RoPE 能正确旋转

        Args:
            x: `(batch, seq_new, d_model)` -- only the NEW tokens, not the whole prefix.
            token_positions: their ABSOLUTE positions, so RoPE rotates by true position
                rather than by offset-within-this-chunk.
            past_kv: `(K, V)` from the previous step, each `(batch, heads, seq_past, d)`,
                or None on prefill.

        Returns:
            `(out, (K, V))` where K/V include both the past and the new tokens.
        """
        # 0. Starting Shape
        x: Float[Tensor, "... seq_new d_model"] # 新的 input x, 可能是一个token: shape=[..., 1, d_model] (decode 阶段), 也可能是多个token: shape=[..., n, d_model] (prefill 阶段)
        W_Q: Float[Tensor, "(h d_k) d_model"]
        W_K: Float[Tensor, "(h d_k) d_model"]
        W_V: Float[Tensor, "(h d_v) d_model"]

        # 1. Transform Projection Matrices to multi-head shape
        W_Q = rearrange(self.W_Q, "(h d_k) d_model -> h d_k d_model", h = self.heads_num, d_k = self.d_k)
        W_K = rearrange(self.W_K, "(h d_k) d_model -> h d_k d_model", h = self.heads_num, d_k = self.d_k)
        W_V = rearrange(self.W_V, "(h d_v) d_model -> h d_v d_model", h = self.heads_num, d_v = self.d_v)

        # 2. Compute 新的 input x 的 Q, K, V representation
        Q = einsum(W_Q, x, "h d_k d_model, ... seq d_model -> ... h seq d_k")
        K = einsum(W_K, x, "h d_k d_model, ... seq d_model -> ... h seq d_k")
        V = einsum(W_V, x, "h d_v d_model, ... seq d_model -> ... h seq d_v")

        # 3. 给 Q,K 做 position encoding
        if self.pos_encod is not None and token_positions is not None:
            Q = self.pos_encod(Q, token_positions)
            K = self.pos_encod(K, token_positions)

        # 4. 把新的 K, V 拼接到 past_kv 上, 形成新的 K, V (prefill 阶段 past_kv=None, decode 阶段 past_kv != None)
        if past_kv is not None: # 说明在 decode 阶段
            K = torch.cat([past_kv[0], K], dim=-2)
            V = torch.cat([past_kv[1], V], dim=-2)

        # 5. 重新 reshape Q, K, V to packed shape for attention kernel
        Q_packed = rearrange(Q, "... h seq d_k -> (... h) seq d_k")
        K_packed = rearrange(K, "... h seq d_k -> (... h) seq d_k")
        V_packed = rearrange(V, "... h seq d_v -> (... h) seq d_v")

        # 6. Compute the attention
        # 6.1 没有提供 mask, 那就说明是正常的 prefill 或 decode, 需要根据 Q 和 K 的 seq_len 是否相等来判断是否是 prefill 阶段, 需不需要 causal mask
        if attn_mask is None:
            #   检查 Q 和 K 的 seq_len 是否相等, 如果相等说明是 prefill 阶段, 需要 causal mask
            #   Prefill: Q_seq_len = K_seq_len = prompt_len 
            #   Decode: Q_seq_len = 1 != K_seq_len = prompt_len + past_decode_kv_len + 1 
            is_causal = Q_packed.shape[-2] == K_packed.shape[-2]
            out_packed = self.attention_fn(Q_packed, K_packed, V_packed, is_causal=is_causal)
        # 6.2 如果提供了 mask, 那就说明是 ragged batch decode, 需要用提供的 mask 来做 attention, 而不是用 causal mask
        else:
            # 用 repeat_interleave 来 match 上 对上打包后的 (B*H, seq_q, seq_k),每个 request 的 mask 要连续重复 H 次
            # 调用方给的 mask 是每个 request 一份,形状 (B, seq_q, seq_k), 这个 mask 对所有的 head 都是一样的, 所以要 repeat_interleave H 次, 变成 (B*H, seq_q, seq_k)
            # 然后就可以对的上 scaled_dot_product_attention 中的 attention matrix 的 shape P = QK = (B*H, seq_q, seq_k) 了, 然后再对齐 B*H 来一个 batch (一个一个 Head 的) 进行操作
            mask = attn_mask.to(Q_packed.device).repeat_interleave(self.heads_num, dim=0)
            out_packed = self.attention_fn(
                Q_packed, K_packed, V_packed, is_causal=False, attn_mask=mask
            )
        
        # 7. Reshape back to (B, H, Seq, d_v)
        out = rearrange(out_packed, "(b h) seq d_v -> b h seq d_v", h=self.heads_num)
        out = rearrange(out, "... h seq d_v -> ... seq (h d_v)")
        out = einsum(self.W_O, out, "d_model (h_d_v), ... seq (h_d_v) -> ... seq d_model")
        return out, (K, V)


    def forward_decode_kvcache(self, x, token_positions, k_cache, v_cache, cache_seqlens, cache_batch_idx=None,
                               impl: str = "auto"):
        """One decode token per row against a PRE-ALLOCATED cache (see kv_cache_attention.py). Inference only.

        x: `(B, 1, d_model)`; token_positions: `(B, 1)` absolute positions (= current lengths, for RoPE);
        k_cache / v_cache: `(n_slots, max_len, H, d)` -- this layer's slice of the batch's cache, written IN PLACE;
        cache_seqlens / cache_batch_idx: `(B,)` int32 (lengths before this token, slot of each row).
        Returns `(B, 1, d_model)`. Nothing is concatenated and no CPU scalar is created: the only per-layer
        work is the projections, RoPE, one cache-append + attention kernel, and the output projection.
        """
        from cs336_basics.transfromer.kv_cache_attention import kvcache_attention

        W_Q = rearrange(self.W_Q, "(h d_k) d_model -> h d_k d_model", h=self.heads_num, d_k=self.d_k)
        W_K = rearrange(self.W_K, "(h d_k) d_model -> h d_k d_model", h=self.heads_num, d_k=self.d_k)
        W_V = rearrange(self.W_V, "(h d_v) d_model -> h d_v d_model", h=self.heads_num, d_v=self.d_v)
        Q = einsum(W_Q, x, "h d_k d_model, ... seq d_model -> ... h seq d_k")
        K = einsum(W_K, x, "h d_k d_model, ... seq d_model -> ... h seq d_k")
        V = einsum(W_V, x, "h d_v d_model, ... seq d_model -> ... h seq d_v")
        if self.pos_encod is not None and token_positions is not None:
            Q = self.pos_encod(Q, token_positions)
            K = self.pos_encod(K, token_positions)
        q = Q.transpose(1, 2).contiguous()        # (B, 1, H, d)
        k_new = K.transpose(1, 2).contiguous()
        v_new = V.transpose(1, 2).contiguous()
        out = kvcache_attention(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, impl=impl)
        out = rearrange(out, "... seq h d_v -> ... seq (h d_v)")
        return einsum(self.W_O, out, "d_model h_d_v, ... seq h_d_v -> ... seq d_model")


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
