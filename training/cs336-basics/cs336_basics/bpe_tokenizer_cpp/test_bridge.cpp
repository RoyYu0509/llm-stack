// test_bridge.cpp — CLI bridge for Python test harness
//
// Usage:
//   ./test_bridge train <input_file> <vocab_size> <vocab_out> <merges_out>
//   ./test_bridge encode <vocab_path> <merges_path> <text>
//   ./test_bridge decode <vocab_path> <merges_path> <id1,id2,id3,...>
//   ./test_bridge roundtrip <vocab_path> <merges_path> <text>
//
// Output is machine-parseable: one line of comma-separated IDs (encode),
// or raw text (decode), or "MATCH"/"MISMATCH" (roundtrip).

#include "bpe_tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// ============================================================================
// Implement load() here since it's TODO in bpe_tokenizer.cpp
// ============================================================================
static void load_tokenizer(BPETokenizer& tok,
                           const std::string& vocab_path,
                           const std::string& merges_path) {
    // We need direct access to internals, but they're private.
    // Instead, we reconstruct by:
    //   1. Creating a fresh tokenizer (has 256 bytes + special tokens)
    //   2. Reading merges file and applying them in order
    //      (this rebuilds the vocab identically to training)

    // Read merges: each line is "hex_bt1 hex_bt2"
    std::ifstream mf(merges_path);
    if (!mf.is_open()) {
        std::cerr << "Error: cannot open merges file: " << merges_path << std::endl;
        return;
    }

    // We can't directly set internals, so we use a different approach:
    // Read vocab to verify, but actually just feed merges to encode/decode.
    // The tokenizer already has init_vocab() called in constructor.
    // We need to call load() which is TODO — but we can work around this
    // by training on a dummy and then replacing.
    //
    // Actually, the cleanest approach: just use tok.load() after we implement it.
    // Since load() is a TODO, let's parse the files and call tok.load().
    tok.load(vocab_path, merges_path);
}

// Helper: parse hex string to raw bytes
static std::string hex_to_bytes(const std::string& hex) {
    std::string result;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        unsigned char byte = static_cast<unsigned char>(
            std::stoi(hex.substr(i, 2), nullptr, 16));
        result.push_back(static_cast<char>(byte));
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_bridge <command> [args...]" << std::endl;
        return 1;
    }

    std::string command = argv[1];
    std::vector<std::string> special_tokens = {"<|endoftext|>"};

    if (command == "train") {
        // train <input_file> <vocab_size> <vocab_out> <merges_out>
        if (argc < 6) {
            std::cerr << "Usage: test_bridge train <input_file> <vocab_size> <vocab_out> <merges_out>" << std::endl;
            return 1;
        }
        std::string input_file = argv[2];
        int vocab_size = std::stoi(argv[3]);
        std::string vocab_out = argv[4];
        std::string merges_out = argv[5];

        BPETokenizer tokenizer(vocab_size, special_tokens);
        tokenizer.train(input_file, vocab_size);
        tokenizer.save(vocab_out, merges_out);

        // Output vocab size and merge count for verification
        std::cout << "VOCAB_SIZE=" << tokenizer.vocab_size() << std::endl;
        std::cout << "NUM_MERGES=" << tokenizer.merges().size() << std::endl;

    } else if (command == "encode") {
        // encode <vocab_path> <merges_path> <text>
        if (argc < 5) {
            std::cerr << "Usage: test_bridge encode <vocab_path> <merges_path> <text>" << std::endl;
            return 1;
        }
        std::string vocab_path = argv[2];
        std::string merges_path = argv[3];

        // Remaining args joined as text (to handle spaces)
        std::string text;
        for (int i = 4; i < argc; i++) {
            if (i > 4) text += " ";
            text += argv[i];
        }

        BPETokenizer tokenizer(10000, special_tokens);
        tokenizer.load(vocab_path, merges_path);

        auto ids = tokenizer.encode(text);

        // Output as comma-separated IDs
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << ids[i];
        }
        std::cout << std::endl;

    } else if (command == "decode") {
        // decode <vocab_path> <merges_path> <id1,id2,...>
        if (argc < 5) {
            std::cerr << "Usage: test_bridge decode <vocab_path> <merges_path> <id1,id2,...>" << std::endl;
            return 1;
        }
        std::string vocab_path = argv[2];
        std::string merges_path = argv[3];
        std::string ids_str = argv[4];

        // Parse comma-separated IDs
        std::vector<int> ids;
        std::istringstream ss(ids_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                ids.push_back(std::stoi(token));
            }
        }

        BPETokenizer tokenizer(10000, special_tokens);
        tokenizer.load(vocab_path, merges_path);

        std::string decoded = tokenizer.decode(ids);
        // Use a long sentinel that won't appear in real decoded text.
        std::cout << decoded << "<<__END_DECODE__>>" << std::endl;

    } else if (command == "roundtrip") {
        // roundtrip <vocab_path> <merges_path> <text>
        if (argc < 5) {
            std::cerr << "Usage: test_bridge roundtrip <vocab_path> <merges_path> <text>" << std::endl;
            return 1;
        }
        std::string vocab_path = argv[2];
        std::string merges_path = argv[3];

        std::string text;
        for (int i = 4; i < argc; i++) {
            if (i > 4) text += " ";
            text += argv[i];
        }

        BPETokenizer tokenizer(10000, special_tokens);
        tokenizer.load(vocab_path, merges_path);

        auto ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);

        // Output IDs
        std::cerr << "IDS=";
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << ids[i];
        }
        std::cerr << std::endl;

        // Output match status
        if (text == decoded) {
            std::cout << "MATCH" << std::endl;
        } else {
            std::cout << "MISMATCH" << std::endl;
            std::cerr << "Original:  [" << text << "]" << std::endl;
            std::cerr << "Decoded:   [" << decoded << "]" << std::endl;
        }

    } else if (command == "encode_stdin") {
        // encode_stdin <vocab_path> <merges_path>
        // Reads text from stdin (supports newlines, special chars)
        if (argc < 4) {
            std::cerr << "Usage: test_bridge encode_stdin <vocab_path> <merges_path>" << std::endl;
            return 1;
        }
        std::string vocab_path = argv[2];
        std::string merges_path = argv[3];

        // Read all of stdin
        std::string text((std::istreambuf_iterator<char>(std::cin)),
                         std::istreambuf_iterator<char>());

        BPETokenizer tokenizer(10000, special_tokens);
        tokenizer.load(vocab_path, merges_path);

        auto ids = tokenizer.encode(text);
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << ids[i];
        }
        std::cout << std::endl;

    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }

    return 0;
}
