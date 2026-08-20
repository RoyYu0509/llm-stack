#include "bpe_tokenizer.h"
#include <iostream>
#include <chrono>

// ============================================================================
// pretokenize — 对应 Python 的 pretokenization()
// ============================================================================
/*
文件/字符串
    │
    ▼
split_by_special_tokens()
    │
    ├─ special token ──→ ByteSeq{ "<EOS>" }（单元素，整体保留）
    │
    └─ 普通文本 ──→ simple_split()
                        │
                        └─ word ──→ ByteSeq{ "h","e","l","l","o" } (Default UTF-8 Byte Level)
                                        │
                                        ▼
                                  pretok_dict_[seq]++
*/
// ============================================================================
void BPETokenizer::pretokenizeSerial(const std::string& file_path) {
    pretok_dict_.clear(); // clear 现在存过的 pretok_dict_

    std::ifstream file(file_path, std::ios::binary); // 用 binary 的形式来读 text, 这样可以直接处理 raw bytes
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << file_path << std::endl;
        return;
    }

    // 按行读取文件 (等同于 Python 的 for line in f:)
    std::string line;
    int64_t line_count = 0;

    auto start_time = std::chrono::steady_clock::now(); // 计时

    while (std::getline(file, line)) { // 一行一行读
        line_count++;

        // Step 1: 把文本按 special token（如 <EOS>）分割成若干 parts (ie. [textchunk1], [<EOS>], [textchunk2], ...)
        std::vector<std::string> parts;
        split_by_special_tokens(line, parts);

        // Step 2: 对每个 part 进行处理
        for (const auto& part : parts) { 
            if (part.empty()) continue;

            // 2.1 先检查这个 part 是不是一个 special token
            bool is_special = false;
            for (const auto& sp : special_tokens_) { // 逐个 special token 检查
                if (part == sp) {
                    is_special = true;
                    break;
                }
            }
            // 如果是 special token, 直接把它作为一个整体加入 pretok_dict_ 计数
            // 同时 continue, 不跑后续的code
            if (is_special) {
                // Special token 作为一个整体
                ByteSeq seq = { part };
                pretok_dict_[seq]++;
                continue;
            }

            // 2.2 part 是普通文本: 那就正常用 regex 来 generate splitting, 然后生成 pretokens = { byte1 : count1, byte2 : count2, ... }
            // 这里用的都是 basic byte UTF-8, 每个 word 的每个 char 单独变成一个 string(1, c)，push 进 ByteSeq
            std::vector<std::string> words = simple_split(part);
            for (const auto& word : words) {
                if (word.empty()) continue;
                
                ByteSeq seq {}; // 创建这个 word 的 pretoken sequence (每个 char 变成一个 byte)
                for (unsigned char c : word) {
                    seq.push_back(std::string(1, static_cast<char>(c)));
                }
                pretok_dict_[seq]++; // 把这个 pretoken sequence 加入 pretok_dict_ 并且 count +1
            }
        }

        // 打印进度
        if (line_count % 10000 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            std::cout << "Processed " << line_count << " lines ("
                      << elapsed << "s)" << std::endl;
        }
    }

    std::cout << "Pretokenization done. " << line_count << " lines, "
              << pretok_dict_.size() << " unique pretokens." << std::endl;
}

// ---------------------------------------------------------------------------
// Helper: 按 special tokens 分割文本
// ---------------------------------------------------------------------------
void BPETokenizer::split_by_special_tokens(const std::string& text,
                                           std::vector<std::string>& parts) const {
    if (special_tokens_.empty()) {
        parts.push_back(text);
        return;
    }

    // 从头扫描, 每次找最早出现的 special token
    size_t pos = 0;
    while (pos < text.size()) {
        // 找最近的 special token
        size_t best_pos = std::string::npos;  // npos ≈ 结尾 (没找到)
        size_t best_len = 0;

        for (const auto& sp : special_tokens_) {
            size_t found = text.find(sp, pos);
            if (found != std::string::npos && (found < best_pos)) {
                best_pos = found;
                best_len = sp.size();
            }
        }

        if (best_pos == std::string::npos) {
            // 没找到更多 special token, 剩余部分全部加入
            parts.push_back(text.substr(pos));
            break;
        }

        // 加入 special token 之前的普通文本
        if (best_pos > pos) {
            parts.push_back(text.substr(pos, best_pos - pos));
        }
        // 加入 special token 本身
        parts.push_back(text.substr(best_pos, best_len));
        pos = best_pos + best_len;
    }
}

// ---------------------------------------------------------------------------
// Helper: 简化分词 — 按空格和常见标点切分
// ---------------------------------------------------------------------------
std::vector<std::string> BPETokenizer::simple_split(const std::string& text) const {
    // 策略: 遇到空格时开始新 word, 但空格本身附加到下一个 word 的开头
    // 这模拟了 GPT-2 pattern 中 " ?\p{L}+" 把前导空格包含进 token 的行为
    std::vector<std::string> result;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            // 如果当前 word 非空, 先结束它
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            // 空白字符自己作为一个 token (或附到下一个 word)
            current.push_back(static_cast<char>(c));
        } else {
            current.push_back(static_cast<char>(c));
        }
    }

    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

// ============================================================================
// text_to_pretokens — 对应 Python 的 _text_2_pretoken_iterator()
// ============================================================================
std::vector<ByteSeq> BPETokenizer::text_to_pretokens(const std::string& text) const {
    std::vector<ByteSeq> result;

    // Step 1: 按 special tokens 分割
    std::vector<std::string> parts;
    split_by_special_tokens(text, parts);

    for (const auto& part : parts) {
        if (part.empty()) continue;

        // 检查是不是 special token
        bool is_special = false;
        for (const auto& sp : special_tokens_) {
            if (part == sp) {
                is_special = true;
                break;
            }
        }

        if (is_special) {
            // Special token 保留为一个整体
            result.push_back({ part });
            continue;
        }

        // 普通文本: 分词后每个 word 拆成单字节
        std::vector<std::string> words = simple_split(part);
        for (const auto& word : words) {
            if (word.empty()) continue;
            ByteSeq seq;
            for (unsigned char c : word) {
                seq.push_back(std::string(1, static_cast<char>(c)));
            }
            result.push_back(seq);
        }
    }
    return result;
}
