#include "bpe_tokenizer.h"
#include "parallel_helper.h" // 需要访问 find_chunk_boundaries() 和 process_chunk()
#include <iostream>   // std::cout, std::cerr
#include <sstream>    // std::istringstream — 用来按行读取字符串
#include <chrono>     // 计时用
#include <thread>    // std::thread — C++ 的线程
/**
 * @brief Constructor for BPETokenizer
 * 
 * @param max_vocab_size Maximum vocabulary size for the tokenizer
 * @param special_tokens A list of special tokens to be included in the vocabulary
 * 
 * 1. 存 Special Tokens 进 Member data
 * 2. 初始化 vocabulary (256 bytes + special tokens) — 调用 init_vocab()
 */
BPETokenizer::BPETokenizer(int max_vocab_size,
                           const std::vector<std::string>& special_tokens)
    : max_vocab_size_(max_vocab_size)        // 初始化列表 ≈ Python self.xxx = xxx
    , special_tokens_(special_tokens)
{
    // 把 special tokens 的字符串形式存一份 (在 C++ 中 string 本身就是 bytes)
    // Python 中你要 .encode("utf-8"), C++ 的 std::string 默认就是 byte sequence
    for (const auto& tok : special_tokens_) {
        special_tokens_bytes_.push_back(tok);
    }

    // 初始化基础 vocab (256 bytes + special tokens)
    init_vocab();
}


//Pretokenization - Parallelized Version
/**
 * @brief Pretokenization on the input file_path
 * 
 * @return void
 */
void BPETokenizer::pretokenizeParallel(const std::string& file_path) {
    pretok_dict_.clear();

    int num_threads = 4;  // 可以改成参数, 或用 std::thread::hardware_concurrency()
    auto start_time = std::chrono::steady_clock::now();

    // Step 1: 计算文件分块边界
    std::vector<size_t> boundaries = find_chunk_boundaries(file_path, num_threads);

    std::cout << "Splitting file into " << (boundaries.size() - 1) << " chunks" << std::endl;

    // Step 2: 每个 thread 处理一个 chunk
    // 关键: 每个 thread 写自己的 local_dict, 不共享, 不需要锁
    std::vector<PretokDict> local_dicts(boundaries.size() - 1);
    std::vector<std::thread> threads;

    for (size_t i = 0; i + 1 < boundaries.size(); i++) {
        // std::thread 的构造函数:
        //   第一个参数: 要执行的函数 (lambda)
        //   lambda 内部捕获需要的变量
        //
        // [&, i] 的含义:
        //   & = 所有外部变量按引用捕获
        //   i = 但 i 按值捕获 (因为循环变量 i 会变, 我们需要每个 thread 拿到自己的 i 值)
        threads.emplace_back([&, i]() {
            local_dicts[i] = process_chunk(
                file_path,
                boundaries[i],
                boundaries[i + 1],
                special_tokens_,
                *this
            );
        });
    }

    // Step 3: 等待所有 thread 完成
    // thread.join() = "等这个 thread 跑完再继续"
    for (auto& t : threads) {
        t.join();
    }

    // Step 4: 合并所有 local_dict 到 pretok_dict_
    for (const auto& local_dict : local_dicts) {
        for (const auto& [seq, count] : local_dict) {
            pretok_dict_[seq] += count;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Parallel pretokenization done in " << elapsed << "s, "
              << pretok_dict_.size() << " unique pretokens." << std::endl;
}


/**
 * @brief Initialize the byte-to-id dictionary with the 256 possible byte values
 *        and (optionally) several special tokens.
 */
void BPETokenizer::init_vocab() {
    // std::unordered_map<std::string, int> byte_2_id_ {};
    // 不应该在这里 init 一个 local 的 dict, 应该直接用 constructor init 好的 member data
    int id {0};

    // 同时创建 byte -> id 和 id -> byte
    // 1. 添加 special token, 如果有的话
    if (!special_tokens_.empty()){
        for (const auto& st : special_tokens_){
            id_2_bytes_[id] = st;
            byte_2_id_[st] = id;
            id++;
        }
    }
    // 2. 添加 default bytes
    for (int j = 0;j < 256;j++) {
        std::string byte_sequence { std::string(1, static_cast<char>(j)) };

        id_2_bytes_[id] = byte_sequence;
        byte_2_id_[byte_sequence] = id;
        
        id++;
    }   
}


/**
 * 遍历 pretok_dict_，统计所有相邻 pair 的频率，存进 freq_dict_。
 * 
 * Member Data:
 *  pretok_dict_: std::unordered_map<ByteSeq, int64_t>，其中 ByteSeq 是 std::vector<std::string>
 *  freq_dict_: std::unordered_map<BytePair, int64_t>，其中 BytePair 是 std::pair<std::string, std::string>
 */
void BPETokenizer::build_freq_dict() {
    // starting with an empty freq_dict
    freq_dict_.clear();

    // 重新 build 新的 frequency Dict
    for (const auto& [seq, count] : pretok_dict_){ // 因为 map 里面是一个一个 pair
        int seq_len {static_cast<int>(seq.size())};
        if (seq_len >= 2){
            for (int i = 1; i < seq_len; i++) {
                BytePair pair {seq[i-1], seq[i]};
                freq_dict_[pair] += count;
            }
        }
    }
}

/**
 * 
 * 在 freq_dict_ 里找频率最高的 pair（频率相同取字典序更大的）
 * 在 pretok_dict_ 里把所有出现该 pair 的地方合并
 * 把合并后的新 bytes 加入 vocab（byte_2_id_ 和 id_2_bytes_）
 * 调用 build_freq_dict() 重建频率表
 * 
 * @return 被 merged 的 BytePair
 */
BytePair BPETokenizer::merge_once() {
    // Step 1: 找频率最高的 pair
    BytePair max_byte_pair {};
    int64_t max_count {};
    for (const auto& [byte_pair, count] : freq_dict_) {
        // (Max Count) OR (If both MAX Count, then update the lexigraphically greater one)
        if (count > max_count || (count == max_count  && byte_pair > max_byte_pair)){
            max_byte_pair = byte_pair;
            max_count = count;
        }
    }
    
    auto [bt0, bt1] = max_byte_pair;
    std::string merged_new_byte {bt0 + bt1};


    // Step 2: 重新 build pretok_dict_
    PretokDict new_pretok_dict_ {}; // 新建 pretok_dict_

    for (const auto& [byte_seq, count] : pretok_dict_) { // 重新 process pretok_dict_ 中的每一个 [byte_seq: count]
        int byte_seq_len {static_cast<int>(std::ssize(byte_seq))};
        ByteSeq new_seq {};

        int i {0};
        while (i < byte_seq_len){
            if (i+1 < byte_seq_len && byte_seq[i] == bt0 && byte_seq[i+1] == bt1){
                new_seq.push_back(merged_new_byte);
                i+=2; // 额外进1
            }
            else {
                new_seq.push_back(byte_seq[i]);
                i++;
            }
        }
        new_pretok_dict_[new_seq] += count; // 把重新 process 的 [byte_seq: count] 添加到新的 new_pretok_dict_ 中
    }
    pretok_dict_ = std::move(new_pretok_dict_); // 用 move 来更新 member data 中的 pretok_dict_


    // Step 3: 加入 vocab (更新 id_2_bytes_ 和 byte_2_id_)
    int64_t next_id {std::ssize(id_2_bytes_)};
    id_2_bytes_[next_id] = merged_new_byte;
    byte_2_id_[merged_new_byte] = next_id;
    

    // Step 4: 用新的 Vocabulary 重新建 freq_dict
    build_freq_dict();

    return max_byte_pair; // return new merged BytePair
}


// ============================================================================
// train — 对应 Python 的 train()
// ============================================================================
std::vector<BytePair> BPETokenizer::train(const std::string& input_path, int vocab_size, bool use_parallel) {
    // 0. 先把之前的 merge_sequence_ 给清除, 因为现在要重新训练
    merge_sequence_.clear();

    // 1. Process `text_file` 来 create pretoken_dict 
    // (这样后面就不用再看 text_file, 直接用 equivalent 的 pretoken_dict 就行)
    if (use_parallel) {
        pretokenizeParallel(input_path);
    } else {
        pretokenizeSerial(input_path);
    }

    // 2. 新建 freq_dict
    build_freq_dict();

    // 3. 重复 merge 直到 reach target Vocabulary size
    while (std::ssize(id_2_bytes_) < vocab_size) {
        merge_sequence_.push_back(merge_once());
    }

    return merge_sequence_; 
}



/**
 * 拿到一个 raw pretoken ByteSeq, 比如 {"t", "h", "e"}, 按 merge_sequence_ 的顺序依次扫描合并.
 * 
 * @return Return a List representation of the pretok_sequence after apply merging.
 */
ByteSeq BPETokenizer::apply_merges(const ByteSeq& pretoken) const {
    // Recoding the token sequence by applying merges
    ByteSeq prev_seq {pretoken}; // create seq copy
    for (const BytePair& merged_pair : merge_sequence_){
        ByteSeq new_seq {};

        const auto& [bt0, bt1] = merged_pair;
        std::string merged_bytes {bt0 + bt1};

        int i {0};
        int seq_len = std::ssize(prev_seq);
        while (i < seq_len) {
            // Case1: merge byte pair
            if (i+1 < seq_len && prev_seq[i] == bt0 && prev_seq[i+1] == bt1){
                new_seq.push_back(merged_bytes);
                i += 2;
            }
            // Case2: regular byte appending
            else {
                new_seq.push_back(prev_seq[i]);
                i++;
            }
        }

        // 记录当前 apply 过 merged_pair 的 byte sequence, 然后继续下个 iter 的 applying merged_pair
        prev_seq = std::move(new_seq);
    }

    return prev_seq;
}

// ============================================================================
// encode — 对应 Python 的 encoding()
// ============================================================================
std::vector<int> BPETokenizer::encode(const std::string& text) const {
    // id sequence
    std::vector<int> id_seq {};
    
    // 把 text process 成很多个 pretoken sequences, 然后存进 std::vector 里
    std::vector<ByteSeq> pretokens {text_to_pretokens(text)};

    // 在每个 pretoken sequence 的内部, 去 apply 所有的 merges
    for (const ByteSeq& pretoken : pretokens) {
        ByteSeq merged_pretoken {apply_merges(pretoken)};
        
        // 把 merged 过的 pretoken sequence 转成 id sequence
        for (const std::string& byte : merged_pretoken){
            // Note: 这里必须要用 find(), 有一套自己的逻辑, 很简单
            // 因为 [] 操作符不是 const 的（因为 key 不存在时它会插入新元素, 而我们的 func 是 const function
            auto it {byte_2_id_.find(byte)};  
            
            if (it != byte_2_id_.end()) // 如果iterator没有出界, 就说明找到了这个 byte 的 hash
            { 
                id_seq.push_back(it->second);  
            }
            else // 说明 current byte 不在我们的 Vocabulary 里
            { 
                id_seq.push_back(-1); // 代表 undefined token id
            }
        }
    }

    return id_seq;
}

/**
 * 拿到一个 std::vector<int> 的 token ids，逐个查 id_2_bytes_
 * 
 * @return the corresponding byte sequence
 */
std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out_bytes {};
    for (int id : ids){
        // 从 id_2_bytes_ 中找对应的 byte
        auto it = id_2_bytes_.find(id);
        if (it != id_2_bytes_.end()){
            out_bytes += it->second;
        }
        else{ // 未知id, 不在 Vocabulary 中; 比如前面 encode 时填的 -1
            out_bytes += "\xEF\xBF\xBD"; // (UTF-8 的 replacement character)
        }
    }
    return out_bytes;
}

// ============================================================================
// save / load — 存 Vocabulary 和 merge_sequence pickle files
// ============================================================================
void BPETokenizer::save(const std::string& vocab_path,
                        const std::string& merges_path) const {
    // --- 保存 vocab: 每行 "id hex_bytes" ---
    {
        std::ofstream out(vocab_path);
        for (const auto& [id, bytes] : id_2_bytes_) {
            out << id << " ";
            // 把 bytes 转成 hex 表示, 方便存储和可读
            for (unsigned char c : bytes) {
                char buf[4];
                snprintf(buf, sizeof(buf), "%02x", c);
                out << buf;
            }
            out << "\n";
        }
    }

    // --- 保存 merges: 每行 "hex_bt1 hex_bt2" ---
    {
        std::ofstream out(merges_path);
        for (const auto& [bt1, bt2] : merge_sequence_) {
            // bt1 转 hex
            for (unsigned char c : bt1) {
                char buf[4];
                snprintf(buf, sizeof(buf), "%02x", c);
                out << buf;
            }
            out << " ";
            // bt2 转 hex
            for (unsigned char c : bt2) {
                char buf[4];
                snprintf(buf, sizeof(buf), "%02x", c);
                out << buf;
            }
            out << "\n";
        }
    }

    std::cout << "Saved vocab to " << vocab_path << std::endl;
    std::cout << "Saved merges to " << merges_path << std::endl;
}

// load — 读取 save() 写出的格式, 重建 byte_2_id_, id_2_bytes_, merge_sequence_
void BPETokenizer::load(const std::string& vocab_path,
                        const std::string& merges_path) {
    // Helper lambda: hex string -> raw bytes
    auto hex_to_bytes = [](const std::string& hex) -> std::string {
        std::string result;
        for (size_t i = 0; i + 1 < hex.size(); i += 2) {
            unsigned char byte = static_cast<unsigned char>(
                std::stoi(hex.substr(i, 2), nullptr, 16));
            result.push_back(static_cast<char>(byte));
        }
        return result;
    };

    // --- 读取 vocab: 每行 "id hex_bytes" ---
    byte_2_id_.clear();
    id_2_bytes_.clear();
    {
        std::ifstream in(vocab_path);
        if (!in.is_open()) {
            std::cerr << "Error: Cannot open vocab file: " << vocab_path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int id;
            std::string hex_str;
            iss >> id >> hex_str;
            std::string bytes = hex_to_bytes(hex_str);
            byte_2_id_[bytes] = id;
            id_2_bytes_[id] = bytes;
        }
    }

    // --- 读取 merges: 每行 "hex_bt1 hex_bt2" ---
    merge_sequence_.clear();
    {
        std::ifstream in(merges_path);
        if (!in.is_open()) {
            std::cerr << "Error: Cannot open merges file: " << merges_path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string hex_bt1, hex_bt2;
            iss >> hex_bt1 >> hex_bt2;
            merge_sequence_.push_back({hex_to_bytes(hex_bt1), hex_to_bytes(hex_bt2)});
        }
    }

    std::cerr << "Loaded vocab (" << byte_2_id_.size() << " entries) and "
              << merge_sequence_.size() << " merges." << std::endl;
}
