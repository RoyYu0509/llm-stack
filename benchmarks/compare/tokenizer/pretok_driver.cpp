// pretok_driver.cpp -- time ONLY the pretokenization stage of the C++ BPE trainer.
//
// This driver does not modify any tokenizer source.
//
// Why a separate driver: `main_benchmark.cpp` times the whole `train()` (pretokenize + all merges), but the
// question here is "how much faster is the C++ pretokenizer than the Python one", and the merge loop (serial,
// full recompute) would swamp it. `pretokenizeSerial` / `pretokenizeParallel` are private members, so the
// driver includes the header with `private` redefined as `public`. All standard headers are included first,
// so only the tokenizer's own class definition sees the macro. The class layout is unchanged in practice
// (GCC/Clang do not reorder by access); the library .cpp files are compiled normally.
//
// Usage:  pretok_driver <corpus> <serial|parallel>
// Output: the tokenizer's own progress lines, then one final line
//         RESULT {"mode":..,"seconds":..,"unique":..,"total":..,"checksum":".."}
//   unique   = number of distinct pretokens (size of pretok_dict_)
//   total    = sum of all counts
//   checksum = order-independent hash of (bytes-of-pretoken, count) pairs, same definition as
//              pretok_stats() in bench_pretokenize.py, so the Python reference can be compared bit-for-bit.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define private public
#include "bpe_tokenizer.h"
#undef private

static uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <corpus> <serial|parallel>" << std::endl;
        return 2;
    }
    const std::string path = argv[1];
    const std::string mode = argv[2];
    if (mode != "serial" && mode != "parallel") {
        std::cerr << "mode must be serial or parallel" << std::endl;
        return 2;
    }

    BPETokenizer tok(512, {"<|endoftext|>"});

    auto t0 = std::chrono::steady_clock::now();
    if (mode == "serial") {
        tok.pretokenizeSerial(path);
    } else {
        tok.pretokenizeParallel(path);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    uint64_t unique = tok.pretok_dict_.size();
    uint64_t total = 0;
    uint64_t checksum = 0;
    for (const auto& [seq, count] : tok.pretok_dict_) {
        std::string joined;
        for (const auto& piece : seq) joined += piece;
        total += static_cast<uint64_t>(count);
        checksum += fnv1a64(joined) ^ (static_cast<uint64_t>(count) * 0x9E3779B97F4A7C15ULL);
    }

    std::cout << "RESULT {\"mode\":\"" << mode << "\",\"seconds\":" << seconds << ",\"unique\":" << unique
              << ",\"total\":" << total << ",\"checksum\":\"" << checksum << "\"}" << std::endl;
    return 0;
}
