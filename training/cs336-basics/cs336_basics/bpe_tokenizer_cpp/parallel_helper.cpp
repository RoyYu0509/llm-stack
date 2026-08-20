// ============================================================================
// 并行化 pretokenize — 需要新增的 #include
// ============================================================================
// 在 bpe_tokenizer.cpp 顶部加上:
#include <thread>    // std::thread — C++ 的线程
#include <mutex>     // std::mutex — 互斥锁 (这里其实不需要, 因为每个 thread 写自己的局部变量)
#include "bpe_tokenizer.h"  // 需要访问 BPETokenizer 的成员函数和数据
#include <fstream>   // std::ifstream — 读文件
#include <iostream>   // std::cout, std::cerr
#include <sstream>    // std::istringstream — 用来按行读取字符串
#include <chrono>     // 计时用

// ============================================================================
// Helper: 找到 splitting 成很多 chunk 的 boundary indices
// 思路: 把文件按字节均分成 N 块, 然后调整到 next <EOS> 位置, 避免切断一个 text body
// ============================================================================
std::vector<size_t> find_chunk_boundaries(
    const std::string& file_path,
    int num_chunks)
{
    // 打开文件, 获取文件大小
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    // std::ios::ate = "at end", 打开后光标在文件末尾, 这样 tellg() 直接得到文件大小
    size_t file_size = file.tellg();

    // 计算均匀分块的初始边界
    std::vector<size_t> boundaries;
    size_t chunk_size = file_size / num_chunks;

    for (int i = 0; i <= num_chunks; i++) {
        if (i == 0) {
            boundaries.push_back(0);
        } else if (i == num_chunks) {
            boundaries.push_back(file_size);
        } else {
            // 预估splitting位置
            size_t pos = i * chunk_size;

            // 从预估splitting位置, 往后找到最近的换行符, 避免切断一行
            file.seekg(pos);   // 移动读取光标到 pos
            std::string temp;
            std::getline(file, temp);  // 读到行末
            pos = file.tellg();        // 现在光标在下一行开头

            // 如果超过文件末尾, 就用文件末尾
            if (pos == std::string::npos || pos >= file_size) {
                pos = file_size;
            }
            boundaries.push_back(pos);
        }
    }
    return boundaries;
}


// ============================================================================
// Helper: 处理文件的一个 chunk
//
// 每个 thread 调用这个函数, 独立处理自己的那块文件, 返回局部 pretok_dict
// 注意: 这个函数不修改任何共享数据, 所以不需要锁
// ============================================================================
PretokDict process_chunk(
    const std::string& file_path,
    size_t start,
    size_t end,
    const std::vector<std::string>& special_tokens,
    const BPETokenizer& tokenizer) // 传入 tokenizer 是为了调用它的 split_by_special_tokens() 和 simple_split()
{
    PretokDict local_dict;

    std::ifstream file(file_path, std::ios::binary);
    file.seekg(start);    // 跳到 chunk 起点

    std::string line;
    while (file.tellg() < static_cast<std::streampos>(end) && std::getline(file, line)) {
        // 和单线程版本完全一样的处理逻辑, 只是写入 local_dict 而不是 pretok_dict_
        std::vector<std::string> parts;
        tokenizer.split_by_special_tokens(line, parts);

        for (const auto& part : parts) {
            if (part.empty()) continue;

            bool is_special = false;
            for (const auto& sp : special_tokens) {
                if (part == sp) { is_special = true; break; }
            }

            if (is_special) {
                ByteSeq seq = { part };
                local_dict[seq]++;
                continue;
            }

            auto words = tokenizer.simple_split(part);
            for (const auto& word : words) {
                if (word.empty()) continue;
                ByteSeq seq;
                for (unsigned char c : word) {
                    seq.push_back(std::string(1, static_cast<char>(c)));
                }
                local_dict[seq]++;
            }
        }
    }

    return local_dict;
}


