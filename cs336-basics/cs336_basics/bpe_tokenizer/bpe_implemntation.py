"""
最小 BPE Tokenizer — 从零开始，每一步都解释
"""

# =============================================================================
# 第一步: 理解数据表示
# =============================================================================
# BPE 操作的对象是 BYTES，不是字符。
# 任何文本都可以变成 bytes:
#   "hello" → [104, 101, 108, 108, 111]  (每个是 0-255 的整数)
#   "你好"  → [228, 189, 160, 229, 165, 189]  (UTF-8 编码，每个中文字 3 bytes)
#
# 起始 vocab 就是 256 个单字节: {b'\x00': 0, b'\x01': 1, ..., b'\xff': 255}
# 这意味着: 不管什么文本，我们都能编码，只是效率低（一个 byte 一个 token）


# =============================================================================
# 第二步: Pretokenization — 把文本切成小块
# =============================================================================
# 为什么要 pretokenize?
#   如果直接对整个文本统计 byte pair，"the" 末尾的 "e" 和下一个词 " cat"
#   开头的 " " 会被统计为一个 pair (e, 空格)，这不合理。
#   所以先按空格/标点把文本切成独立的小块 (pretokens)，
#   每个 pretoken 内部做 pair 统计，pretoken 之间不跨越。

def pretokenize(text: str) -> list[list[bytes]]:
    """
    把文本切成 pretokens，每个 pretoken 是一个 list of single bytes。
    
    简化版: 按空格切分 (GPT-2 用复杂的 regex，原理一样)
    
    输入: "the cat the"
    输出: [
        [b't', b'h', b'e'],           # "the"
        [b' ', b'c', b'a', b't'],     # " cat" (空格属于下一个词)
        [b' ', b't', b'h', b'e'],     # " the"
    ]
    """
    # 简化分词: 第一个词单独，后面的词带前导空格
    words = text.split(' ')
    pretokens = []
    for i, word in enumerate(words):
        if i > 0:
            word = ' ' + word  # 前导空格附到词上
        # 拆成单字节
        single_bytes = [bytes([b]) for b in word.encode('utf-8')]
        pretokens.append(single_bytes)
    return pretokens


# =============================================================================
# 第三步: 建立 pretok_dict — 统计每个 pretoken 出现了几次
# =============================================================================
# 为什么要 pretok_dict?
#   文本里 "the" 可能出现 5000 次，我们不需要存 5000 份，
#   只需要存一份 (b't', b'h', b'e') 和它的 count 5000。

def build_pretok_dict(text: str) -> dict[tuple, int]:
    """
    输入: "the cat the"
    输出: {
        (b't', b'h', b'e'):        1,   # "the" 出现 1 次
        (b' ', b'c', b'a', b't'):  1,   # " cat" 出现 1 次
        (b' ', b't', b'h', b'e'):  1,   # " the" 出现 1 次
    }
    
    如果输入是 "the the the cat":
    输出: {
        (b't', b'h', b'e'):        1,   # 第一个 "the"
        (b' ', b't', b'h', b'e'):  2,   # " the" 出现 2 次
        (b' ', b'c', b'a', b't'):  1,
    }
    """
    pretokens = pretokenize(text)
    pretok_dict = {}
    for pretoken in pretokens:
        key = tuple(pretoken)  # list 不能做 dict key, 转成 tuple
        pretok_dict[key] = pretok_dict.get(key, 0) + 1
    return pretok_dict


# =============================================================================
# 第四步: 建立 freq_dict — 统计相邻 pair 的频率
# =============================================================================
# 从 pretok_dict 里统计: 哪两个相邻的 bytes 最经常一起出现?

def build_freq_dict(pretok_dict: dict) -> dict[tuple, int]:
    """
    输入 pretok_dict: {
        (b't', b'h', b'e'):        1,
        (b' ', b't', b'h', b'e'):  2,
        (b' ', b'c', b'a', b't'):  1,
    }
    
    输出 freq_dict: {
        (b't', b'h'): 1 + 2 = 3,   ← "the"贡献1, " the"贡献2
        (b'h', b'e'): 1 + 2 = 3,   ← 同上
        (b' ', b't'): 2,            ← " the"贡献2
        (b' ', b'c'): 1,
        (b'c', b'a'): 1,
        (b'a', b't'): 1,
    }
    """
    freq_dict = {}
    for byte_seq, count in pretok_dict.items():
        for i in range(len(byte_seq) - 1):
            pair = (byte_seq[i], byte_seq[i + 1])
            freq_dict[pair] = freq_dict.get(pair, 0) + count
    return freq_dict


# =============================================================================
# 第五步: Merge — 合并最高频 pair
# =============================================================================
# 找到频率最高的 pair，在所有 pretoken 中合并它

def merge_in_pretok_dict(pretok_dict: dict, pair: tuple) -> dict:
    """
    把 pretok_dict 中所有出现 pair 的地方合并。
    
    例: pair = (b't', b'h')
    
    pretok_dict 变化:
        (b't', b'h', b'e')       → (b'th', b'e')
        (b' ', b't', b'h', b'e') → (b' ', b'th', b'e')
        (b' ', b'c', b'a', b't') → 不变 (没有 t,h 相邻)
    """
    bt1, bt2 = pair
    merged = bt1 + bt2  # b't' + b'h' = b'th'
    
    new_pretok_dict = {}
    for byte_seq, count in pretok_dict.items():
        new_seq = []
        i = 0
        while i < len(byte_seq):
            if i + 1 < len(byte_seq) and byte_seq[i] == bt1 and byte_seq[i + 1] == bt2:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(byte_seq[i])
                i += 1
        new_pretok_dict[tuple(new_seq)] = new_pretok_dict.get(tuple(new_seq), 0) + count
    
    return new_pretok_dict


# =============================================================================
# 第六步: Training — 把上面的步骤串起来
# =============================================================================

def train(text: str, target_vocab_size: int):
    """
    完整 BPE 训练流程。
    
    返回:
        vocab:  dict[int, bytes]     id -> bytes 的映射
        merges: list[tuple]          merge 历史记录
    """
    # --- 初始化 vocab: 256 个单字节 ---
    vocab = {i: bytes([i]) for i in range(256)}  # {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
    byte_2_id = {bytes([i]): i for i in range(256)}
    next_id = 256
    
    # --- 建 pretok_dict ---
    pretok_dict = build_pretok_dict(text)
    
    print("=== Initial pretok_dict ===")
    for seq, cnt in pretok_dict.items():
        readable = [s.decode('utf-8', errors='replace') for s in seq]
        print(f"  {readable}: {cnt}")
    
    # --- Merge loop ---
    merges = []
    
    while len(vocab) < target_vocab_size:
        # 建频率表
        freq_dict = build_freq_dict(pretok_dict)
        
        if not freq_dict:
            print("No more pairs to merge!")
            break
        
        # 找最高频 pair
        best_pair = max(freq_dict, key=lambda p: (freq_dict[p], p))
        best_freq = freq_dict[best_pair]
        
        # 合并
        merged_bytes = best_pair[0] + best_pair[1]
        pretok_dict = merge_in_pretok_dict(pretok_dict, best_pair)
        
        # 更新 vocab
        vocab[next_id] = merged_bytes
        byte_2_id[merged_bytes] = next_id
        merges.append(best_pair)
        
        print(f"Merge #{len(merges)}: {best_pair[0]} + {best_pair[1]} -> {merged_bytes}  (freq={best_freq}, id={next_id})")
        next_id += 1
    
    return vocab, byte_2_id, merges


# =============================================================================
# 第七步: Encode — 用训练好的 merges 编码新文本
# =============================================================================
# 注意: encode 也要先 pretokenize！
#   这样每个 pretoken 独立地 apply merges，不会跨词合并。

def apply_merges(pretoken: list[bytes], merges: list[tuple]) -> list[bytes]:
    """
    对一个 pretoken 按顺序应用所有 merges。
    
    例: pretoken = [b't', b'h', b'e']
        merges = [(b't', b'h'), (b'th', b'e')]
    
    第一轮 merge (b't', b'h'):
        [b't', b'h', b'e'] → [b'th', b'e']
    
    第二轮 merge (b'th', b'e'):
        [b'th', b'e'] → [b'the']
    
    返回: [b'the']
    """
    current = pretoken
    
    for bt1, bt2 in merges:
        new_seq = []
        i = 0
        while i < len(current):
            if i + 1 < len(current) and current[i] == bt1 and current[i + 1] == bt2:
                new_seq.append(bt1 + bt2)
                i += 2
            else:
                new_seq.append(current[i])
                i += 1
        current = new_seq
        
        if len(current) <= 1:
            break  # 已经合并成一个 token, 不需要继续
    
    return current


def encode(text: str, byte_2_id: dict, merges: list) -> list[int]:
    """
    编码文本为 token IDs。
    
    步骤:
    1. pretokenize: "the cat" → [["t","h","e"], [" ","c","a","t"]]
    2. 对每个 pretoken 调用 apply_merges
    3. 查 byte_2_id 得到 id
    """
    pretokens = pretokenize(text)  # ← 这里也要 pretokenize！
    
    result = []
    for pretoken in pretokens:
        merged = apply_merges(pretoken, merges)
        for token_bytes in merged:
            result.append(byte_2_id[token_bytes])
    
    return result


# =============================================================================
# 第八步: Decode — 简单地查表拼回来
# =============================================================================

def decode(ids: list[int], vocab: dict) -> str:
    """
    把 token IDs 拼回文本。
    直接查 vocab 表，把 bytes 拼起来，decode 成字符串。
    """
    byte_chunks = []
    for id in ids:
        byte_chunks.append(vocab[id])
    return b"".join(byte_chunks).decode("utf-8", errors="replace")


# =============================================================================
# 运行示例
# =============================================================================

if __name__ == "__main__":
    text = "the cat the dog the cat the the dog"
    target_vocab_size = 262  # 256 + 6 次 merge
    
    print(f"Input text: \"{text}\"\n")
    
    # --- Train ---
    vocab, byte_2_id, merges = train(text, target_vocab_size)
    
    # --- Encode ---
    print(f"\n=== Encode Test ===")
    test = "the cat"
    ids = encode(test, byte_2_id, merges)
    print(f"Text:    \"{test}\"")
    print(f"Tokens:  {ids}")
    print(f"Count:   {len(test.encode('utf-8'))} bytes → {len(ids)} tokens")
    
    # --- Decode ---
    decoded = decode(ids, vocab)
    print(f"Decoded: \"{decoded}\"")
    print(f"Match:   {test == decoded}")
    
    # --- 显示 vocab ---
    print(f"\n=== Learned Vocab (beyond 256 bytes) ===")
    for id in range(256, len(vocab)):
        print(f"  id={id}: {vocab[id]}")
