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
#include "bpe_tokenizer.h"  // 需要访问 BPETokenizer 的成员函数和数据

// Declare Helper Functions
std::vector<size_t> find_chunk_boundaries(const std::string& file_path, int num_chunks);
PretokDict process_chunk(const std::string& file_path, size_t start, size_t end, const std::vector<std::string>& special_tokens, const BPETokenizer& tokenizer);

