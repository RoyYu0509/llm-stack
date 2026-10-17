"""
管理「所有正在 decode 的请求」共用的一份 KV cache。

- 布局:         每行的真实 key 靠右放,左边补零。lengths[b] 既是第 b 行的真实长度,也是它下一个 token 的 RoPE 位置。
- 补齐方法:      每步的新 K/V 永远 cat 在最右边一列,右对齐时每行的真实 key 永远是连续的一段,新 key 紧挨着接上;右补齐的话,新 key 会落在 padding 后面,mask 要跳过中间的空洞。
- 添加req:      新行比批窄,就给新行左补零;新行比批宽,就给已有各行左补零。
- 移除req:      index_select 留下其余行,再裁掉「对剩下所有行都是 padding」的最左几列。
- batch_step:  一次前向,positions = lengths,mask = [每行的真实区域 | 新 token 自己];跑完 lengths 加 1。
- Cost:        每次加入、移出都要把整批 KV 拷贝一遍(torch.cat / index_select),显存还随最长那行按 padding 浪费

"""

from typing import Any, Dict, Hashable, List, Optional, Tuple
import torch
import torch.nn as nn
from cs336_basics.lm import TransformerLM

KV = List[Tuple[torch.Tensor, torch.Tensor]]

class RaggedDecodeBatch:
    """
    Left-padded, per-layer KV cache for every request that is currently DECODING, 
    so one `forward_with_cache` call advances all of them by one token.

    # Invariants:
    -------------
    1. row b's real key slots are the RIGHT-most `lengths[b]` columns; everything to their
      left is padding (zeros) and is hidden by the attention mask;

    2. `lengths[b]` is also the absolute position of row b's next token (RoPE), because the
       cache width batched_kv_len includes padding and therefore is NOT a position;

    3. a new column is appended for every row on each step, so right-alignment is preserved;

    4. rows leave by index_select, then all-padding leading columns are trimmed.
    """
    def __init__(self, model: TransformerLM|nn.Module) -> None:
        self.model: TransformerLM|nn.Module = model
        self.keys: List[Hashable] = []               # self.keys = ['R0', 'R1', 'R2']   谁在批里,按行顺序
        self.past: Optional[KV] = None               # self.past = [(k1, v1), (k2, v2), (k3, v3)]   每个请求的 KV cache
        self.lengths: Optional[torch.Tensor] = None  # self.lengths = [6, 2, 9]         每个请求的真实 token 数

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.keys

    @property
    def width(self) -> int:
        """ 返回当前 batch 的 KV cache 的 sequence 长度, 也就是 KV cache 的列数 """
        return 0 if self.past is None else self.past[0][0].shape[-2]

    @staticmethod
    def _left_pad(kv: KV, n: int) -> KV:
        """ 对 K, V matrices 的左边添加 n 个 padding """
        if n == 0:
            return kv
        out = []
        for k, v in kv:
            pk = k.new_zeros(k.shape[0], k.shape[1], n, k.shape[3])
            pv = v.new_zeros(v.shape[0], v.shape[1], n, v.shape[3])
            out.append((torch.cat([pk, k], dim=-2), torch.cat([pv, v], dim=-2)))

        return out

    def add(self, key: Hashable, row_past: KV) -> None:
        """ 
        添加一个 new_request 到 batch 中, 将 new_request 和 已经有的 batch 的 KV cache 拼起来, 形成新的 batch KV cache.
        这些 KV cache 都是 (B, H, batched_kv_len, d) 形状的矩形张量, 在添加 new_request 之后必须要保证 所有的 B 个 KV cache 的 batched_kv_len 都是一样的, 所以要做 padding.
        """
        # 检查是不是已经在里面了
        if key in self.keys:
            raise KeyError(f"row {key!r} is already in the batch")

        # 新请求的
        new_seq_len = row_past[0][0].shape[-2]
        device = row_past[0][0].device

        # 如果 batch 还没有任何请求 (keys 里面没有存任何的 request key), 就直接把这个请求的 KV cache 作为 batch 的 KV cache
        if not self.keys:
            self.past = [(k, v) for k, v in row_past]
            self.lengths = torch.tensor([new_seq_len], dtype=torch.long, device=device)
            self.keys = [key]
            return
        # 如果 batch 中已经有请求了, 就要把这个请求的 KV cache 和已有的 KV cache 拼起来, 并且要重新做 padding, 确保所有的 KV Cache 长度一致
        else:
            # 重新做 padding 来对齐 dimension, 让所有 KV 堆在一起变成 (B, H, T, d) 形状的矩形张量
            batched_kv_len = self.width
            existing, new = self.past, row_past
            
            if new_seq_len > batched_kv_len:
                existing = self._left_pad(existing, new_seq_len - batched_kv_len)
            elif new_seq_len < batched_kv_len:
                new = self._left_pad(new, batched_kv_len - new_seq_len)
            self.past = [
                (torch.cat([ek, nk], dim=0), torch.cat([ev, nv], dim=0))
                for (ek, ev), (nk, nv) in zip(existing, new)
            ]

            self.lengths = torch.cat([self.lengths, torch.tensor([new_seq_len], dtype=torch.long, device=device)])
            self.keys.append(key)


    def remove(self, key: Hashable) -> None:
        """ 移除一个请求, 并且把剩下的请求的 KV cache 拼起来, 形成新的 batch KV cache. """
        # 找到要移除的请求的索引, 并且记录要保留的请求的索引
        i = self.keys.index(key)  # 找到这个 key 在 keys 里面的 index
        keep = [j for j in range(len(self.keys)) if j != i]  # 记录其他要保留的请求的索引
        self.keys.pop(i)  # 移除 key 这个 request

        # 如果没有剩下的请求了, 就把 batch 的 KV cache 清空
        if not keep:
            self.past, self.lengths = None, None
            return
        # 如果还有剩下的请求, 就把剩下的请求的 KV cache 拼起来, 形成新的 batch KV cache. 并且要重新做 padding, 确保所有的 KV Cache 长度一致
        else:
            # 0. 创建一个索引张量, 用来挑选剩下的请求
            keep_idx = torch.tensor(keep, dtype=torch.long, device=self.lengths.device)

            # 1: 只保留还在 batch 里的那些行(沿 batch 维挑选)
            kept_past = []
            for k, v in self.past:  # self.past 是 list, 每一项是一层的 (K, V), 我们现在要遍历每一层 KV
                # 每一层拿要保留的 KV (keep_idx 指定的行)
                kept_k = k.index_select(0, keep_idx) 
                kept_v = v.index_select(0, keep_idx)
                # 把这一层要保留的 KV 加入到 kept_past 里面
                kept_past.append((kept_k, kept_v))

            # 2: 算出来现在需要的 KV cache 的宽度, 只考虑还在 batch 里的那些行的 lengths
            kept_lengths = self.lengths.index_select(0, keep_idx)
            old_width = kept_past[0][0].shape[-2]
            new_width = int(kept_lengths.max())

            # 3: 如果可以 trim 的话, 就把左边不需要的 padding trim 掉
            if new_width < old_width:
                trim = old_width - new_width
                trimmed_past = []
                for k, v in kept_past:
                    trimmed_k = k[:, :, trim:, :]
                    trimmed_v = v[:, :, trim:, :]
                    trimmed_past.append((trimmed_k, trimmed_v))
                kept_past = trimmed_past

            # 4: 存回 batch
            self.past = kept_past
            self.lengths = kept_lengths


    def valid_mask(self) -> torch.Tensor:
        """
        返回一个 (B, batched_kv_len) 的 per-request mask, 用来标记每行的真实 key 的位置.
        
        每行的真实 key 的位置是 True, padding 的位置是 False. 
        """
        batched_kv_len = self.width  # batch KV tensor sequence 长度
        cols = torch.arange(batched_kv_len, device=self.lengths.device)  # (batched_kv_len,) [0,1,...,batched_kv_len-1]
        threshold = batched_kv_len - self.lengths  # 把每一行的真 token 的起始位置算出来
        return cols.unsqueeze(0) >= threshold.unsqueeze(1) # 最后进行 boardcast, 返回一个 (B, batched_kv_len) 的 mask, 每行的真实 key 的位置是 True (也就是大于那行的 threshold), padding 的位置是 False.


    def step(self, tokens: Dict[Hashable, int]) -> Dict[Hashable, torch.Tensor]:
        """ 对 Batch 中的所有请求进行一次前向, 每个请求都喂入一个 input token, 返回一个 logits. """
        # input tokens 须和 batch 里的请求一模一样,不能多也不能少
        if set(tokens) != set(self.keys):
            raise KeyError(f"decode step got rows {sorted(map(str, tokens))}, batch has {sorted(map(str, self.keys))}")

        device = self.lengths.device
        B = len(self.keys)

        # 1. 把所有 request 的 input token 拼成一个 (B, 1) 的张量, 每行是一个请求的 token
        x = torch.tensor([[int(tokens[k])] for k in self.keys], dtype=torch.long, device=device)
        positions = self.lengths.unsqueeze(1)   # lengths 是「每行已有多少个 token」,所以它同时就是「下一个 token 的位置」

        # 2. 生成一个 (B, 1, batched_kv_len+1) 的 attention mask, 用来标记每行的真实 key 的位置和新 token 自己的位置.
        """
        valid_mask(), shape=(B, batched_kv_len)    (3, 9)       哪些旧槽位是真的
        一列 ones, shape=(B, 1) True                (3, 10)      新 token 自己的 k 也要被看到
        unsqueeze(1)                               (3, 1, 10)   插入 seq_q=1 这一维
        """
        mask = torch.cat(
            [
                self.valid_mask(), 
                torch.ones(B, 1, dtype=torch.bool, device=device)
            ], 
            dim=1,
        ).unsqueeze(1)

        # 3. 调用 model.forward_with_cache, 喂入所有请求的 input token, 返回一个 (B, 1, vocab_size) 的 logits 和新的 KV cache
        logits, new_past = self.model.forward_with_cache(
            x, past_kv=self.past, positions=positions, attn_mask=mask
        )

        self.past = new_past
        self.lengths = self.lengths + 1
        return {k: logits[i, -1, :] for i, k in enumerate(self.keys)}
