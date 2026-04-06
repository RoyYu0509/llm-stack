# BPE Tokenizer 完整流程 — 逐步推演

## 起点: Text Corpus

```
text = "the the the cat the cat"
```

---

## Phase 1: Training

### Step 1 — Pretokenize

把文本按空格切分成 pretokens。注意：除了第一个词，每个词的前导空格会附到词上。

```
text = "the the the cat the cat"
  ↓ 按空格切分
pretokens = ["the", " the", " the", " cat", " the", " cat"]
```

每个 pretoken 拆成单字节：

```
pretokens = [
    pretoken_0 = [b"t", b"h", b"e"]          ← "the"
    pretoken_1 = [b" ", b"t", b"h", b"e"]    ← " the"
    pretoken_2 = [b" ", b"t", b"h", b"e"]    ← " the"
    pretoken_3 = [b" ", b"c", b"a", b"t"]    ← " cat"
    pretoken_4 = [b" ", b"t", b"h", b"e"]    ← " the"
    pretoken_5 = [b" ", b"c", b"a", b"t"]    ← " cat"
]
```

### Step 2 — 建 pretok_dict（统计每个 pretoken 出现几次）

相同的 pretoken 合并计数，不需要存多份：

```
pretok_dict = {
    (b"t", b"h", b"e"):        1    ← pretoken_0 出现 1 次
    (b" ", b"t", b"h", b"e"):  3    ← pretoken_1,2,4 相同，合并计数
    (b" ", b"c", b"a", b"t"):  2    ← pretoken_3,5 相同，合并计数
}
```

### Step 3 — 建 freq_dict（统计相邻 pair 频率）

遍历 pretok_dict 的每个 entry，看内部有哪些相邻 pair，乘以 count：

```
来自 pretok_dict[(b"t", b"h", b"e")] = 1:
    freq_dict[(b"t", b"h")] += 1
    freq_dict[(b"h", b"e")] += 1

来自 pretok_dict[(b" ", b"t", b"h", b"e")] = 3:
    freq_dict[(b" ", b"t")] += 3
    freq_dict[(b"t", b"h")] += 3
    freq_dict[(b"h", b"e")] += 3

来自 pretok_dict[(b" ", b"c", b"a", b"t")] = 2:
    freq_dict[(b" ", b"c")] += 2
    freq_dict[(b"c", b"a")] += 2
    freq_dict[(b"a", b"t")] += 2
```

汇总：

```
freq_dict = {
    (b"t", b"h"): 4    ← 最高！
    (b"h", b"e"): 4    ← 并列最高
    (b" ", b"t"): 3
    (b" ", b"c"): 2
    (b"c", b"a"): 2
    (b"a", b"t"): 2
}
```

### Step 4 — 第一次 Merge

```
best_pair = (b"t", b"h")     ← 频率最高，平局取字典序更大
best_freq = 4
merged_bytes = b"t" + b"h" = b"th"
new_id = 256
```

**更新 vocab**:

```
byte_2_id[b"th"] = 256
id_2_bytes[256]  = b"th"
```

**更新 pretok_dict**（扫描每个 entry，把相邻的 b"t", b"h" 合并成 b"th"）：

```
pretok_dict[(b"t", b"h", b"e")] = 1:
    i=0: b"t",b"h" 匹配 best_pair → 合并
    new_seq = (b"th", b"e")

pretok_dict[(b" ", b"t", b"h", b"e")] = 3:
    i=0: b" ",b"t" 不匹配 → 保留 b" "
    i=1: b"t",b"h" 匹配 → 合并
    new_seq = (b" ", b"th", b"e")

pretok_dict[(b" ", b"c", b"a", b"t")] = 2:
    没有 b"t",b"h" 相邻 → 不变
    new_seq = (b" ", b"c", b"a", b"t")
```

新的 pretok_dict:

```
pretok_dict = {
    (b"th", b"e"):             1
    (b" ", b"th", b"e"):       3
    (b" ", b"c", b"a", b"t"):  2
}
```

**记录 merge**:

```
merge_sequence = [merge_0 = (b"t", b"h")]
```

### Step 5 — 重建 freq_dict

```
来自 pretok_dict[(b"th", b"e")] = 1:
    freq_dict[(b"th", b"e")] += 1

来自 pretok_dict[(b" ", b"th", b"e")] = 3:
    freq_dict[(b" ", b"th")] += 3
    freq_dict[(b"th", b"e")] += 3

来自 pretok_dict[(b" ", b"c", b"a", b"t")] = 2:
    freq_dict[(b" ", b"c")] += 2
    freq_dict[(b"c", b"a")] += 2
    freq_dict[(b"a", b"t")] += 2
```

汇总：

```
freq_dict = {
    (b"th", b"e"): 4    ← 最高！
    (b" ", b"th"): 3
    (b" ", b"c"):  2
    (b"c", b"a"):  2
    (b"a", b"t"):  2
}
```

### Step 6 — 第二次 Merge

```
best_pair = (b"th", b"e")
best_freq = 4
merged_bytes = b"th" + b"e" = b"the"
new_id = 257
```

**更新 vocab**:

```
byte_2_id[b"the"] = 257
id_2_bytes[257]   = b"the"
```

**更新 pretok_dict**:

```
pretok_dict[(b"th", b"e")] = 1:
    i=0: b"th",b"e" 匹配 → 合并
    new_seq = (b"the",)

pretok_dict[(b" ", b"th", b"e")] = 3:
    i=0: b" ",b"th" 不匹配 → 保留 b" "
    i=1: b"th",b"e" 匹配 → 合并
    new_seq = (b" ", b"the")

pretok_dict[(b" ", b"c", b"a", b"t")] = 2:
    不变
    new_seq = (b" ", b"c", b"a", b"t")
```

新的 pretok_dict:

```
pretok_dict = {
    (b"the",):              1
    (b" ", b"the"):         3
    (b" ", b"c", b"a", b"t"): 2
}
```

**记录 merge**:

```
merge_sequence = [merge_0 = (b"t", b"h"), merge_1 = (b"th", b"e")]
```

### Step 7 — 重建 freq_dict

```
来自 pretok_dict[(b"the",)] = 1:
    长度只有 1，没有相邻 pair，跳过

来自 pretok_dict[(b" ", b"the")] = 3:
    freq_dict[(b" ", b"the")] += 3

来自 pretok_dict[(b" ", b"c", b"a", b"t")] = 2:
    freq_dict[(b" ", b"c")] += 2
    freq_dict[(b"c", b"a")] += 2
    freq_dict[(b"a", b"t")] += 2
```

```
freq_dict = {
    (b" ", b"the"): 3    ← 最高！
    (b" ", b"c"):   2
    (b"c", b"a"):   2
    (b"a", b"t"):   2
}
```

### Step 8 — 第三次 Merge

```
best_pair = (b" ", b"the")
best_freq = 3
merged_bytes = b" " + b"the" = b" the"
new_id = 258
```

**更新 vocab**:

```
byte_2_id[b" the"] = 258
id_2_bytes[258]    = b" the"
```

**更新 pretok_dict**:

```
pretok_dict[(b"the",)] = 1:
    不变（没有 b" " 和 b"the" 相邻）
    new_seq = (b"the",)

pretok_dict[(b" ", b"the")] = 3:
    i=0: b" ",b"the" 匹配 → 合并
    new_seq = (b" the",)

pretok_dict[(b" ", b"c", b"a", b"t")] = 2:
    i=0: b" ",b"c" → b" " 匹配但 b"c" ≠ b"the" → 不匹配
    不变
    new_seq = (b" ", b"c", b"a", b"t")
```

新的 pretok_dict:

```
pretok_dict = {
    (b"the",):              1
    (b" the",):             3
    (b" ", b"c", b"a", b"t"): 2
}
```

**记录 merge**:

```
merge_sequence = [merge_0 = (b"t", b"h"), merge_1 = (b"th", b"e"), merge_2 = (b" ", b"the")]
```

**训练在这里停止** (假设 vocab_size = 259)。

---

## 最终训练产物

### vocab (只列新增的)

```
id_2_bytes = {
    0: b'\x00', 1: b'\x01', ..., 255: b'\xff',    ← 256 个单字节
    256: b"th",
    257: b"the",
    258: b" the"
}
```

### merge_sequence (顺序很重要！encoding 时要按这个顺序重放)

```
merge_sequence = [
    merge_0 = (b"t", b"h"),       ← 第一个被合并的 pair
    merge_1 = (b"th", b"e"),      ← 第二个
    merge_2 = (b" ", b"the"),     ← 第三个
]
```

---

## Phase 2: Encoding

```
input_text = "the cat"
```

### Step 1 — Pretokenize（和 training 时用同样的切分规则）

```
input_text = "the cat"
  ↓ 按空格切分，拆成单字节
pretokens = [
    pretoken_0 = [b"t", b"h", b"e"]          ← "the"
    pretoken_1 = [b" ", b"c", b"a", b"t"]    ← " cat"
]
```

### Step 2 — 对每个 pretoken 依次 apply merge_sequence

**处理 pretoken_0 = [b"t", b"h", b"e"]**

```
current = [b"t", b"h", b"e"]

Apply merge_0 = (b"t", b"h"):
    i=0: current[0]=b"t", current[1]=b"h" → 匹配！合并
    current = [b"th", b"e"]

Apply merge_1 = (b"th", b"e"):
    i=0: current[0]=b"th", current[1]=b"e" → 匹配！合并
    current = [b"the"]

Apply merge_2 = (b" ", b"the"):
    current 长度只有 1，没有相邻 pair，跳过

pretoken_0_result = [b"the"]
```

**处理 pretoken_1 = [b" ", b"c", b"a", b"t"]**

```
current = [b" ", b"c", b"a", b"t"]

Apply merge_0 = (b"t", b"h"):
    i=0: b" ",b"c" → 不匹配
    i=1: b"c",b"a" → 不匹配
    i=2: b"a",b"t" → 不匹配 (需要 b"t",b"h" 但这里是 b"a",b"t")
    current = [b" ", b"c", b"a", b"t"]    ← 不变

Apply merge_1 = (b"th", b"e"):
    没有 b"th" 出现
    current = [b" ", b"c", b"a", b"t"]    ← 不变

Apply merge_2 = (b" ", b"the"):
    没有 b"the" 出现
    current = [b" ", b"c", b"a", b"t"]    ← 不变

pretoken_1_result = [b" ", b"c", b"a", b"t"]
```

### Step 3 — 查 byte_2_id 得到 token IDs

```
pretoken_0_result = [b"the"]
    byte_2_id[b"the"] = 257
    → ids_0 = [257]

pretoken_1_result = [b" ", b"c", b"a", b"t"]
    byte_2_id[b" "] = 32       ← 空格的 ASCII 值
    byte_2_id[b"c"] = 99
    byte_2_id[b"a"] = 97
    byte_2_id[b"t"] = 116
    → ids_1 = [32, 99, 97, 116]
```

**合并所有 pretoken 的 ids**:

```
token_ids = ids_0 + ids_1 = [257, 32, 99, 97, 116]
```

原始: 7 bytes → 编码后: 5 tokens

---

## Phase 3: Decoding

```
token_ids = [257, 32, 99, 97, 116]
```

逐个查 id_2_bytes:

```
id_2_bytes[257] = b"the"
id_2_bytes[32]  = b" "
id_2_bytes[99]  = b"c"
id_2_bytes[97]  = b"a"
id_2_bytes[116] = b"t"
```

拼接:

```
result_bytes = b"the" + b" " + b"c" + b"a" + b"t" = b"the cat"
result_string = "the cat" ✓
```

---

## 总结: 每个函数做什么

```
Training:
    pretokenize(text)    → text 切成 pretokens → 统计次数 → pretok_dict
    build_freq_dict()    → 扫描 pretok_dict 里的相邻 pair → freq_dict
    merge_once()         → 找 freq_dict 最高频 pair → 更新 pretok_dict → 更新 vocab → 重建 freq_dict
    train()              → pretokenize + build_freq_dict + 循环调 merge_once

Encoding:
    text_to_pretokens(text) → 把新文本切成 pretokens（和 training 时同样的切分规则）
    apply_merges(pretoken)  → 对单个 pretoken，按 merge_sequence 顺序执行合并
    encode(text)            → text_to_pretokens + 对每个 pretoken 调 apply_merges + 查 byte_2_id 得 id

Decoding:
    decode(token_ids)       → 查 id_2_bytes 把每个 id 转回 bytes → 拼接 → 转字符串
```
