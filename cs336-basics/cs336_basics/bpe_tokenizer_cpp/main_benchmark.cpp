#include "bpe_tokenizer.h"
#include <iostream>
#include <chrono>
#include <iomanip>  // std::setw, std::setprecision — 格式化输出

// ============================================================================
// 用法:
//   ./bpe_benchmark <input_text_file> <vocab_size>
//
// 例:
//   ./bpe_benchmark data/TinyStoriesV2-GPT4-train.txt 500
// ============================================================================

struct BenchmarkResult {
    std::string label;
    double pretok_time;    // pretokenization 时间 (秒)
    double train_time;     // 总训练时间 (秒)
    int vocab_size;
    int num_merges;
    bool encode_decode_ok; // roundtrip 测试是否通过
};

// 运行一次完整的 train + encode/decode 测试
BenchmarkResult run_benchmark(
    const std::string& label,
    const std::string& input_file,
    int vocab_size,
    bool use_parallel)
{
    BenchmarkResult result;
    result.label = label;

    std::vector<std::string> special_tokens = { "<|endoftext|>" };
    BPETokenizer tokenizer(vocab_size, special_tokens);

    // 计时: 总训练
    auto t0 = std::chrono::steady_clock::now();
    tokenizer.train(input_file, vocab_size, use_parallel);
    auto t1 = std::chrono::steady_clock::now();

    result.train_time = std::chrono::duration<double>(t1 - t0).count();
    result.vocab_size = tokenizer.vocab_size();
    result.num_merges = static_cast<int>(tokenizer.merges().size());

    // Roundtrip 测试
    std::vector<std::string> test_texts = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "<|endoftext|>Once upon a time there was a little girl<|endoftext|>",
    };

    result.encode_decode_ok = true;
    for (const auto& text : test_texts) {
        auto ids = tokenizer.encode(text);
        auto decoded = tokenizer.decode(ids);
        if (text != decoded) {
            result.encode_decode_ok = false;
            std::cerr << "  FAIL: \"" << text << "\" != \"" << decoded << "\"" << std::endl;
        }
    }

    // 额外: 测试 encode 速度
    // 编码一段较长文本, 测 throughput
    std::string long_text = "";
    for (int i = 0; i < 100; i++) {
        long_text += "Once upon a time there was a little girl named Lily. She loved to play in the garden. ";
    }

    auto e0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        auto ids = tokenizer.encode(long_text);
    }
    auto e1 = std::chrono::steady_clock::now();
    double encode_time = std::chrono::duration<double>(e1 - e0).count();

    std::cout << "  Encode speed: " << std::fixed << std::setprecision(1)
              << (10.0 * long_text.size() / 1024.0) / encode_time << " KB/s" << std::endl;

    return result;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════╦════════════╦════════╦════════╦═════════╗" << std::endl;
    std::cout << "║ Configuration            ║ Train (s)  ║ Vocab  ║ Merges ║ Correct ║" << std::endl;
    std::cout << "╠══════════════════════════╬════════════╬════════╬════════╬═════════╣" << std::endl;

    for (const auto& r : results) {
        std::cout << "║ " << std::left << std::setw(25) << r.label
                  << "║ " << std::right << std::setw(9) << std::fixed << std::setprecision(3) << r.train_time << "s"
                  << " ║ " << std::setw(5) << r.vocab_size
                  << " ║ " << std::setw(5) << r.num_merges << "  "
                  << "║   " << (r.encode_decode_ok ? "YES" : "NO ") << "   ║"
                  << std::endl;
    }

    std::cout << "╚══════════════════════════╩════════════╩════════╩════════╩═════════╝" << std::endl;

    // Speedup 计算
    if (results.size() >= 2) {
        double baseline = results[0].train_time;
        std::cout << "\nSpeedup vs " << results[0].label << ":" << std::endl;
        for (size_t i = 1; i < results.size(); i++) {
            double speedup = baseline / results[i].train_time;
            std::cout << "  " << results[i].label << ": "
                      << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <vocab_size>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    int vocab_size = std::stoi(argv[2]);

    std::cout << "╔══════════════════════════════════════╗" << std::endl;
    std::cout << "║   BPE Tokenizer Benchmark            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════╣" << std::endl;
    std::cout << "║ File:       " << std::left << std::setw(25) << input_file << "║" << std::endl;
    std::cout << "║ Vocab size: " << std::left << std::setw(25) << vocab_size << "║" << std::endl;
    std::cout << "╚══════════════════════════════════════╝" << std::endl;

    std::vector<BenchmarkResult> results;

    // --- Run 1: Serial ---
    std::cout << "\n[1/2] Running serial..." << std::endl;
    results.push_back(run_benchmark("Serial (1 thread)", input_file, vocab_size, false));

    // --- Run 2: Parallel ---
    std::cout << "\n[2/2] Running parallel..." << std::endl;
    results.push_back(run_benchmark("Parallel (4 threads)", input_file, vocab_size, true));

    // --- Results ---
    print_results(results);

    return 0;
}
