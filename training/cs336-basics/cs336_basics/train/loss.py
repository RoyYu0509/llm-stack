from cs336_basics.transfromer.embedding import Embedding
from cs336_basics.transfromer.linear_module import Linear
from cs336_basics.transfromer.multiheads_attention import MultiHeadsAttention
from cs336_basics.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from cs336_basics.transfromer.positionalNencoding import PosEncod
from cs336_basics.transfromer.rmsnorm import Rmsnorm
from cs336_basics.transfromer.transformer import PreNormTransformer
from cs336_basics.transfromer.scaled_dot_prod_attention import softmax
from einops import reduce
from jaxtyping import Float
import torch.nn as nn
import torch
from torch import logsumexp



def cross_entropy(out_logits, target):
    """
    Compute the cross-entropy (negative log-likelihood) on the LM's output

    Inputs:
        - out_logits: [Batch, ..., Seq_len] The output logits for each position of the sentence.
        - target: [Batch, ..., Seq_len, Vocab_dist] The target tokenized sentence.
    
    Output:
        - nll: A scalar loss value
    """
    # 检查 target 的 shape 是否正确
    if len(target.shape) == 1:
        d = target.shape[0]
        seq_l = 1
    else:
        d = target.shape[0]
        seq_l = target.shape[-1]
    
    # 1. 提前减去最大值，做数值平移（防止指数爆炸）
    max_val, _ = torch.max(out_logits, dim=-1, keepdim=True) # Keep dim for Broadcasting
    out_logits = out_logits - max_val

    # 2. 把 target words 的 index 拿出来, 后面通过 gather 来取出对应的 prediction logits
    target_words: Float[torch.Tensor, f"Batch, ..., Seq_len, 1"] = target.unsqueeze(-1)
    out_logits: Float[torch.Tensor, f"Batch, ..., Seq_len, Vocab_dist"]

    # 2. 用 Log-Sum-Exp Trick 展开 Log-Softmax 计算：log( exp(o_target) / sum(exp(o_j)) ) 
    #    数学展开为：o_target - log( sum(exp(o_j)) )
    #    这里的 logsumexp(out_logits) 内部执行的就是 log( sum(exp(out_logits)) )
    nll = (out_logits.gather(-1, target_words)) - (logsumexp(out_logits, dim=-1, keepdim = True))
    loss = 1/(d*seq_l) * -nll.sum()
    return loss


def perplexity(logits, targets):
    """
    Perplexity 是给人类看的指标，用来衡量语言模型的性能。

    困惑度的公式是: PPL = exp( cross_entropy )，也就是交叉熵的指数。
    
    它的物理意义极其直观: PPL 代表模型在预测下一个 Token 时, 平均相当于在“多少个完全等概率的候选词”中做选择。
        PPL = 100: 模型非常困惑, 预测下一个词时, 相当于在 100 个等概率选项中盲猜。
        PPL = 10: 模型比较自信, 相当于缩小到了 10 个选项中做选择。
        PPL = 1: 完美的预测, 模型 100% 确定下一个词是什么（$e^0 = 1$）。
    """
    loss = cross_entropy(logits, targets)
    return torch.exp(loss)
