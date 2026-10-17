from cs336_basics.transfromer.embedding import Embedding
from cs336_basics.transfromer.linear_module import Linear
from cs336_basics.transfromer.multiheads_attention import MultiHeadsAttention
from cs336_basics.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from cs336_basics.transfromer.positionalNencoding import PosEncod
from cs336_basics.transfromer.rmsnorm import Rmsnorm
from cs336_basics.transfromer.transformer import PreNormTransformer
from cs336_basics.transfromer.scaled_dot_prod_attention import scaled_dot_product_attention, softmax
import torch.nn as nn
from cs336_basics import nvtx_compat as nvtx
import torch
from cs336_basics import nvtx_compat as nvtx

class TransformerLM(nn.Module):
    """
    Transformer language model for masked next-token prediction. It works like: 
        - Giving a token sequence sentence: [ID1, ID2, ID3, ...]
        - Embedded into vectors
        - For each position i:
            - Gives all token until i.
            - Predict the distribution of vocabulary logits of the next i+1 token.

    Inputs:
        - sentence: Tensor[Int] with shape (B, T) 
            Input is the batched sentences, each sentence sequence is represented 
            by tokenized ID.

    Outputs:
        - word_logits: Tensor[Float] with shape (B, T, V). 
            For each position t in a sequence, the output logits[:, t, :] contains the 
            unnormalized vocabulary scores for predicting the next token given all tokens
            until t.
        
    Notation:
        B: Batch Szie
        T: Sequence Length
        D: The Embedding(input) vector dimension
        V: The Vocabulary Size
        L: Numbers of Layers
        H: Number of Heads
        d: Dimension of Key and Query vector per Head ≈ D/H
        d_ff: Dimension of the hidden layers in the SwiGLU FNN

    Component:
        1. Token Embedding
        2. RoPE Pos Encoding
        3. Transformer Blocks:
            a. RMSNorms
            b. Multi-Head Attention Projection
            c. Residual Addition
            ---------------------------------
            d. RMSNorms
            e. Point-Wise FNN
            f. Residual Addition
        
        4. Normalization
        5. Output Embedding(Logits)

    
    Computational Costs:
        1. Token Embedding: 
            Token ID = i used to retrieved the vector Embedd_Mat[i] of size D.
            Retrieve using one-hot vector MatMul ≈ B * T * (2VD) FLOPs.

        2. RoPE:
            For each embedded vector, we need to apply a 2by2 rotation matrix on
            all of its components. One 2by2 MatVec Mul costs ≈ 4 + 2 FLOPs
            Each Pos Encode would costs ≈ B * T * H * (d) * (6)

        3. Transformer Block:
            i. RMSNorm:
                For each activation vector, RMS compt costs ≈ 2*D + 1 (Add, Mul, Sqrt)
                Normalization costs: ≈ D
                In total ≈ B*T*(D + 2*D + 1) = 2*B*T*D + B*T*D + B*T

            ii. Attention Block:
                1). QKV Projection: 
                    Compute each projection using W@x.
                    Each projection costs ≈ B * T * H * (2(d)*D) = 2*B*T*D^2
                    In total 3 projection operation ≈ 6*B*T*D^2

                2). Pos Encode QK: 
                    Encode QK, each costs ≈ B * T * H * (d) * (6)
                    In total ≈ 2 * (B * T * H * (d) * (6))

                3). Scaled Attention:
                    For each head one sequence, we compute: 
                    - Scaled MatMul QK/sqrt(d) ≈ 2*T*d*T + 1 FLOPs
                    - SoftMax Transforming for one sequence: O(T)
                    - Retrieve from V with the Attention: 2 * T * T * d
                    In total ≈ B*H*[(2*T*d*T+1)+T+(2*T*T*d)] ≈ B*H*4*T*T*d + B*H + B*H*T
                
                4). Out Projection
                    Each Head costs ≈ 2*B*T*d*D
                    In total ≈ H*(2*B*T*d*D)
            
            iii. Residue Adds:
                Element-wise addition, costs ≈ B*T*T

            iv. RMSNorm: ≈ 2*B*T*D + B*T*D + B*T
            
            v. PointWise FNN (SwiGLU):
                For each activation vector:
                    W1x costs: 2*d_ff*D
                    SiLU Act costs: 2*d_ff
                    Gating costs: d_ff
                    W3x costs: 2*d_ff*D
                    W2x costs: 2*d_ff*D
                In total: B*T*(6*d_ff*D + 3*d_ff)
            
            vi. Residue Adds: ≈ B*T*T
        
        4. RMSNorm:
            Apply RMSNorm on the downstream activation vector, costs ≈ 2*B*T*D + B*T*D + B*T
        
        5. Output Embedding:
            Return the raw logits for all V vocabulary, each downstream vec costs ≈ 2*V*D*1
            In total: ≈ B*T*(2*V*D)
    """
    def __init__(self, 
                 vocab_size: int, context_length: int, num_layers: int, # LM args
                 d_model: int, heads_num: int, # Multi-Head Attention args
                 d_ff: int, # FNN args
                 theta: float, # Encoder para
                 pos_encoder=None, # Multi-Head Attention kwargs
                 eps: float = 1e-5, # rmsnorm kwargs
                 latent_exp_factor = 8/3, # FNN kwargs
                 device=None, dtype=torch.float32,  # general kwargs
                 attention_fn: callable = scaled_dot_product_attention,
                 is_causal: bool = True,
    ):
        """
        A transformer block.

        Parameters:
            vocab_size:     int | The size of the vocabulary, necessary for determining the 
                            dimensionality of the token embedding matrix.

            context_length: int | The maximum context length, necessary for determining 
                            the dimensionality of the position embedding matrix.

            num_layers:     int | The number of Transformer blocks to use.
        """
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        # Input Embedding Block
        self.in_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model,
                                   device=device, dtype=dtype)

        # Positional Encoder
        self.pos_encoder = pos_encoder if pos_encoder is not None else PosEncod(theta, d_model//heads_num, context_length, device=device)
        
        # Defining token position with maximum length 
        # Keep token position buffer on the same device as the model to avoid device mismatches during compile.
        token_positions = torch.arange(0, context_length, device=device, dtype=torch.long)
        self.register_buffer("token_positions", token_positions)
        
        # Transformer Blocks
        self.tf_layers = nn.ModuleList([
            PreNormTransformer(
                d_model, heads_num,
                d_ff,
                pos_encod=self.pos_encoder, token_positions=None,
                latent_exp_factor=latent_exp_factor,
                device = device, dtype=dtype,
                attention_fn=attention_fn,
                is_causal=is_causal,
            ) for _ in range(num_layers)
        ])

        # Normalization Layer
        self.norm = Rmsnorm(d_model, eps, device, dtype)

        # Output Linear Block
        self.head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    supports_ragged_decode = True
    def forward_with_cache(self, x, past_kv=None, positions=None, attn_mask=None, last_only: bool = False,
                           last_index=None):
        """Incremental forward for autoregressive decoding. Inference only.

        `forward` is left untouched -- it is the training path, and this is additive.

        Args:
            x: `(batch, seq_new)` token ids. The whole prompt on prefill, then one token
                per decode step.
            past_kv: list of per-layer `(K, V)` from the previous call, or None.

        Returns:
            `(logits, past_kv)` where logits covers only the tokens passed in (so on decode
            it is `(batch, 1, vocab)`), and past_kv should be fed back on the next step.

        Cost: O(prefix) per step instead of the O(prefix^2) of re-running `forward` over the
        whole sequence each time -- see cs336_systems/experiments/bench_serving.py, which
        measures the difference rather than asserting it.
        """
        # 新的 input 的 shape
        seq_new = x.shape[1]
        # 0. 算当前 step 的 position, 如果没有提供 positions, 就通过 past_len 自己算 position 
        # 0.1 如果没有给 positions, 那就一定是单条 inference, 大概率是 prefill 任务. 通过 past_kv 来算出 past_len, 然后算出来 positions
        if positions is None:
            past_len = 0 if past_kv is None else past_kv[0][0].shape[-2]
            if past_len + seq_new > self.token_positions.shape[0]: # 检查长度有没有超过限制
                raise RuntimeError(
                    f"cached decode would reach position {past_len + seq_new}, beyond the "
                    f"model's context length {self.token_positions.shape[0]}"
                )

            # Absolute positions continuing past the cache, so RoPE stays consistent with what
            # the cached keys were rotated by.
            positions = self.token_positions[past_len:past_len + seq_new]
            positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        # 0.2 如果有提供 positions, 就直接用给的 position, 大概率 batch decode, 不用自己去 compute position 了, 直接用外部传入的
        else:
            # Ragged batch: every row carries its own absolute positions, because rows
            # have different true lengths (the cache width includes padding, so it is NOT a position).
            if int(positions.max()) >= self.token_positions.shape[0]:
                raise RuntimeError(
                    f"cached decode would reach position {int(positions.max()) + 1}, beyond the "
                    f"model's context length {self.token_positions.shape[0]}"
                )

        # 1. Input Embedding
        x = self.in_embedding(x)  

        # 2. 让 input x 通过所有的 Transformer Blocks, return 最终的 logits 和新的 KV Cache
        new_kv = []  # 用来存 每一层的新的 KV Cache

        # 过每一层的 Transformer Block, 并且把每一层的新的 KV Cache 存起来
        positions = positions.to(x.device) # 把 position 放到相同的 device 上
        tf_block: PreNormTransformer
        for i, tf_block in enumerate(self.tf_layers):
            layer_past = None if past_kv is None else past_kv[i]
            # 把 positions 传给每一层的 Transformer Block, 让不同的 request 都能拿到各自的 position, 以便 RoPE 能正确旋转
            x, kv = tf_block.forward_with_cache(
                x, token_positions=positions, past_kv=layer_past, attn_mask=attn_mask
            )
            new_kv.append(kv)

        if last_index is not None:
            # batched prefill of right-padded prompts: row i's last REAL token is at last_index[i], not at -1
            rows = torch.arange(x.shape[0], device=x.device)
            x = x[rows, last_index.to(x.device)].unsqueeze(1)
        elif last_only:
            # prefill only needs the last position's logits: skip the LM head (d_model x vocab) for the other S-1
            # positions and the (1, S, vocab) logits tensor
            x = x[:, -1:, :]
        x = self.norm(x)
        x = self.head(x)
        return x, new_kv


    supports_prealloc_decode = True

    def forward_decode_kvcache(self, tokens, positions, k_caches, v_caches, cache_seqlens, cache_batch_idx=None,
                               impl: str = "auto"):
        """One decode step for B rows against a PRE-ALLOCATED per-layer cache. Inference only.

        tokens: `(B, 1)` ids; positions: `(B, 1)` absolute positions (= each row's current length);
        k_caches / v_caches: per-layer `(n_slots, max_len, H, d)`, appended IN PLACE;
        cache_seqlens / cache_batch_idx: `(B,)` int32 -- lengths before this token and the slot of each row.
        Returns logits `(B, 1, vocab)`. The caller increments the lengths once per step (not once per layer).
        """
        x = self.in_embedding(tokens)
        for i, tf_block in enumerate(self.tf_layers):
            x = tf_block.forward_decode_kvcache(
                x, positions, k_caches[i], v_caches[i], cache_seqlens, cache_batch_idx, impl=impl
            )
        x = self.norm(x)
        return self.head(x)


    def forward(self, x):
        """
        Input:
            - x: A batched sequence of integer token IDs, (batch_size, sequence_length)

        Output:
            - A Tensor of (batch_size, sequence_length, vocab_size) a raw logits distribution over 
            the vocabulary.
        """
        # MatMul: (batch_size, sequence_length) . (vocab_size, embedding_dim) 
        with nvtx.range("LM Input Embedding"):
            x = self.in_embedding(x)
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        positions = self.token_positions[:seq_len].unsqueeze(0).expand(batch_size, -1)
        positions = positions.to(x.device)

        with nvtx.range("LM Transformer Blocks Forward Pass"):
            for tf_block in self.tf_layers:
                x = tf_block(x, token_positions=positions)

        with nvtx.range("LM Final Norm"):
            x = self.norm(x)

        with nvtx.range("LM Output Head"):
            x = self.head(x)
        
        # softmax(x, -1)  # Softmax muted, Return raw logit
        return x
