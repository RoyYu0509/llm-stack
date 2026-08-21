#pragma once
// `#pragma once` 等同于 Python 中 "只 import 一次"
// 防止这个头文件被多次 include 导致重复定义

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>    // uint8_t, int32_t 等固定宽度整数类型
#include <fstream>    // 文件读写: std::ifstream / std::ofstream
#include <utility>    // std::pair

// ============================================================================
// C++ 新概念速查 (对照你已经学过的 Python)
// ============================================================================
//
// std::string          ≈ Python 的 bytes (在这里我们用它存 byte sequence)
// std::vector<T>       ≈ Python 的 list[T] (动态数组)
// std::unordered_map   ≈ Python 的 dict (哈希表, O(1) 查找)
// std::map             ≈ Python 的 dict, 但 key 有序 (红黑树, O(log n))
// std::pair<A,B>       ≈ Python 的 tuple[A, B] (两个元素的元组)
// using / typedef      ≈ Python 的 type alias
// const &              ≈ Python 中传参时不复制对象 (C++ 独有概念)
// ============================================================================

// ---------------------------------------------------------------------------
// 类型别名 — 让后面的代码更易读
// ---------------------------------------------------------------------------

// 一个 pretoken: 由多个 byte-string 组成的序列
//   Python 中是 tuple(b"l", b"o", b"w")
//   C++ 中是 vector<string>{"l", "o", "w"}  (每个 string 是 1+ 个 bytes)
using ByteSeq = std::vector<std::string>;

// Pair of byte-strings — 对应你 Python 中的 (bt1, bt2)
using BytePair = std::pair<std::string, std::string>;

// ---------------------------------------------------------------------------
// 自定义 hash 函数: 让 unordered_map 能用 ByteSeq 做 key
// ---------------------------------------------------------------------------
// Python 的 tuple 天然可 hash，但 C++ 的 vector 不行，需要我们自己写
struct ByteSeqHash {
    size_t operator()(const ByteSeq& seq) const {
        // 简单的哈希: 把每个 string 的 hash 混合起来
        size_t seed = seq.size();
        std::hash<std::string> hasher;
        for (const auto& s : seq) {
            // `^` 是异或, `<<` 是左移 — 这是常见的哈希混合技巧
            seed ^= hasher(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// 同样给 BytePair 写一个 hash
struct BytePairHash {
    size_t operator()(const BytePair& p) const {
        std::hash<std::string> h;
        size_t seed = h(p.first);
        seed ^= h(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// ---------------------------------------------------------------------------
// Pretokenization Dict: ByteSeq -> count
//   等同于你 Python 中的 self.pretok_dict = Counter()
// ---------------------------------------------------------------------------
using PretokDict = std::unordered_map<ByteSeq, int64_t, ByteSeqHash>;

// ---------------------------------------------------------------------------
// Frequency Dict: BytePair -> count
//   等同于你 Python 中的 self.freq_dict = {}
// ---------------------------------------------------------------------------
using FreqDict = std::unordered_map<BytePair, int64_t, BytePairHash>;


// ============================================================================
// BPETokenizer 类 — 对应你 Python 的 BBPE 类
// ============================================================================
class BPETokenizer {
public:
    // -----------------------------------------------------------------------
    // 构造函数 (≈ Python __init__)
    // -----------------------------------------------------------------------
    BPETokenizer(int max_vocab_size, const std::vector<std::string>& special_tokens);

    // -----------------------------------------------------------------------
    // Training — 对应你 Python 的 train()
    // -----------------------------------------------------------------------
    // 返回值: merges 序列
    // 训练完成后 vocab (id_2_bytes) 也会被更新
    std::vector<BytePair> train(const std::string& input_path, int vocab_size, bool use_parallel = true);

    // -----------------------------------------------------------------------
    // Encoding — 对应你 Python 的 encoding()
    // -----------------------------------------------------------------------
    std::vector<int> encode(const std::string& text) const;

    // -----------------------------------------------------------------------
    // Decoding — 对应你 Python 的 decoding()
    // -----------------------------------------------------------------------
    std::string decode(const std::vector<int>& ids) const;

    // -----------------------------------------------------------------------
    // 序列化: 保存 / 加载 vocab 和 merges 到文件
    // (Python 中你用的是 pickle, C++ 中我们用简单的文本格式)
    // -----------------------------------------------------------------------
    void save(const std::string& vocab_path, const std::string& merges_path) const;
    void load(const std::string& vocab_path, const std::string& merges_path);

    // -----------------------------------------------------------------------
    // Getters — 查看内部状态 (面试时可能需要讨论)
    // -----------------------------------------------------------------------
    int vocab_size() const { return static_cast<int>(byte_2_id_.size()); }
    const std::vector<BytePair>& merges() const { return merge_sequence_; }

    // -----------------------------------------------------------------------
    // Helpers - 用来 partition text-level 和 word-level 的逻辑, 让代码更清晰
    // -----------------------------------------------------------------------
    void split_by_special_tokens(const std::string& text,
                                 std::vector<std::string>& parts) const;
    std::vector<std::string> simple_split(const std::string& text) const;

private:
    // -----------------------------------------------------------------------
    // 内部方法 — 对应你 Python 的下划线方法
    // -----------------------------------------------------------------------

    // 初始化 256 个单字节 + special tokens 到 vocab
    // 对应 Python 的 intialize_b2id_dict()
    void init_vocab();
    
    // -----------------------------------------------------------------------
    // Pretokenization 
    // -----------------------------------------------------------------------
    void pretokenizeSerial(const std::string& file_path);
    void pretokenizeParallel(const std::string& file_path);


    // 构建频率字典
    // 对应 Python 的 _build_freq_dict()
    void build_freq_dict();

    // 执行一次 merge 并返回被合并的 pair
    // 对应 Python 的 _merge()
    BytePair merge_once();

    // 把文本切成 pretoken 序列 (单线程, 不用 regex — 阶段一简化版)
    // 对应 Python 的 _text_2_pretoken_iterator()
    std::vector<ByteSeq> text_to_pretokens(const std::string& text) const;

    // 对一个 pretoken 应用 merge 序列
    // 对应 Python 的 _apply_merges()
    ByteSeq apply_merges(const ByteSeq& pretoken) const;

    // -----------------------------------------------------------------------
    // Member Data
    // -----------------------------------------------------------------------
    int max_vocab_size_;

    // vocab: bytes <-> id 双向映射
    std::unordered_map<std::string, int> byte_2_id_;   // bytes -> id
    std::unordered_map<int, std::string> id_2_bytes_;   // id -> bytes

    // pretokenization 计数
    PretokDict pretok_dict_;

    // pair 频率
    FreqDict freq_dict_;

    // special tokens
    std::vector<std::string> special_tokens_;        // 原始字符串形式
    std::vector<std::string> special_tokens_bytes_;   // UTF-8 bytes 形式

    // merge 历史记录
    std::vector<BytePair> merge_sequence_;
};


