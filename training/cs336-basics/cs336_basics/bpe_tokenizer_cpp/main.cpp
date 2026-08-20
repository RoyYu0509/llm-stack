#include "bpe_tokenizer.h"
#include <iostream>
#include <fstream>
#include <chrono>

// ============================================================================
// 用法:
//   ./bpe_train <input_text_file> <vocab_size>
//
// 例:
//   ./bpe_train data/tiny_test.txt 300
// ============================================================================
int main(int argc, char* argv[]) {
    // --- 解析命令行参数 ---
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <vocab_size>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    int vocab_size = std::stoi(argv[2]);  // std::stoi: string to int

    // --- 创建 tokenizer ---
    std::vector<std::string> special_tokens = { "<|endoftext|>" };
    BPETokenizer tokenizer(vocab_size, special_tokens);

    // --- 训练 ---
    std::cout << "=== Training BPE Tokenizer ===" << std::endl;
    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Target vocab size: " << vocab_size << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    auto merges = tokenizer.train(input_file, vocab_size);
    auto t1 = std::chrono::steady_clock::now();

    double total_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Training took " << total_time << "s" << std::endl;
    std::cout << "Final vocab size: " << tokenizer.vocab_size() << std::endl;
    std::cout << "Merges performed: " << merges.size() << std::endl;

    // --- 保存 ---
    tokenizer.save("vocab.txt", "merges.txt");


    std::cout << "Vocab size: " << tokenizer.vocab_size() << std::endl;
    std::cout << "Merges count: " << tokenizer.merges().size() << std::endl;

    // 检查单字节是否都在 vocab 里
    std::string test_byte(1, 'H');
    auto ids = tokenizer.encode("H");
    std::cout << "Single 'H' encodes to: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;

    // --- 测试 encode / decode ---
    std::cout << "\n=== Encode / Decode Test ===" << std::endl;

    std::vector<std::string> test_texts = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "<|endoftext|>Once upon a time<|endoftext|>",
    };

    for (const auto& text : test_texts) {
        auto ids = tokenizer.encode(text);
        auto decoded = tokenizer.decode(ids);

        std::cout << "\nOriginal:  \"" << text << "\"" << std::endl;
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "Decoded:   \"" << decoded << "\"" << std::endl;
        std::cout << "Match:     " << (text == decoded ? "YES" : "NO") << std::endl;
        std::cout << "Bytes: " << text.size()
                  << "  Tokens: " << ids.size()
                  << "  Ratio: " << (ids.empty() ? 0.0 : (double)text.size() / ids.size())
                  << std::endl;
    }

    return 0;
}
