#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

/**
 * C++ BPE tokenizer for NinoGPT (header-only).
 *
 * Token layout:
 *   0-255:     raw byte tokens
 *   256:       <|system|>
 *   257:       <|user|>
 *   258:       <|nino|>
 *   259-1023:  BPE merge tokens
 *
 * Loads from tokenizer.txt produced by bpe_tokenizer.py.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>

static constexpr uint16_t SYSTEM_TOKEN_ID = 256;
static constexpr uint16_t USER_TOKEN_ID   = 257;
static constexpr uint16_t NINO_TOKEN_ID   = 258;
static constexpr uint16_t FIRST_MERGE_ID  = 259;

struct SpecialToken {
    const char* text;
    uint16_t id;
};

static const SpecialToken SPECIAL_TOKENS[] = {
    {"<|system|>", SYSTEM_TOKEN_ID},
    {"<|user|>",   USER_TOKEN_ID},
    {"<|nino|>",   NINO_TOKEN_ID},
};
static constexpr size_t NUM_SPECIAL_TOKENS = 3;

struct BPETokenizer {
    int vocab_size = 0;
    std::vector<std::pair<int,int>> merges;
    std::vector<std::vector<uint8_t>> vocab;  // token_id -> byte sequence

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string line;
        // Line 1: vocab_size
        if (!std::getline(f, line)) return false;
        vocab_size = std::stoi(line);

        // Line 2: num_merges
        if (!std::getline(f, line)) return false;
        int num_merges = std::stoi(line);

        merges.clear();
        merges.reserve(num_merges);
        for (int i = 0; i < num_merges; i++) {
            if (!std::getline(f, line)) return false;
            std::istringstream ss(line);
            int a, b;
            ss >> a >> b;
            merges.push_back({a, b});
        }

        // Build vocab table
        vocab.resize(vocab_size);
        for (int i = 0; i < 256; i++) {
            vocab[i] = {static_cast<uint8_t>(i)};
        }
        // Special tokens get their string representation as bytes
        vocab[SYSTEM_TOKEN_ID] = {'<','|','s','y','s','t','e','m','|','>'};
        vocab[USER_TOKEN_ID]   = {'<','|','u','s','e','r','|','>'};
        vocab[NINO_TOKEN_ID]   = {'<','|','n','i','n','o','|','>'};

        for (int i = 0; i < num_merges; i++) {
            int a = merges[i].first;
            int b = merges[i].second;
            auto& va = vocab[a];
            auto& vb = vocab[b];
            vocab[FIRST_MERGE_ID + i].clear();
            vocab[FIRST_MERGE_ID + i].insert(vocab[FIRST_MERGE_ID + i].end(), va.begin(), va.end());
            vocab[FIRST_MERGE_ID + i].insert(vocab[FIRST_MERGE_ID + i].end(), vb.begin(), vb.end());
        }

        return true;
    }

    std::vector<uint16_t> encode(const std::string& text) const {
        // Step 1: Replace special token strings with their IDs, bytes otherwise
        std::vector<int> ids;
        size_t i = 0;
        while (i < text.size()) {
            bool matched = false;
            for (size_t s = 0; s < NUM_SPECIAL_TOKENS; s++) {
                const char* tok = SPECIAL_TOKENS[s].text;
                size_t tok_len = std::strlen(tok);
                if (i + tok_len <= text.size() && text.compare(i, tok_len, tok) == 0) {
                    ids.push_back(SPECIAL_TOKENS[s].id);
                    i += tok_len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                ids.push_back(static_cast<uint8_t>(text[i]));
                i++;
            }
        }

        // Step 2: Apply merges in order
        for (size_t m = 0; m < merges.size(); m++) {
            int a = merges[m].first;
            int b = merges[m].second;
            int new_id = FIRST_MERGE_ID + static_cast<int>(m);
            std::vector<int> new_ids;
            size_t j = 0;
            while (j < ids.size()) {
                if (j + 1 < ids.size() && ids[j] == a && ids[j+1] == b) {
                    new_ids.push_back(new_id);
                    j += 2;
                } else {
                    new_ids.push_back(ids[j]);
                    j++;
                }
            }
            ids = std::move(new_ids);
        }

        // Convert to uint16_t
        std::vector<uint16_t> result(ids.size());
        for (size_t k = 0; k < ids.size(); k++) {
            result[k] = static_cast<uint16_t>(ids[k]);
        }
        return result;
    }

    std::string decode(const std::vector<uint16_t>& ids) const {
        std::string result;
        for (uint16_t id : ids) {
            if (id < static_cast<uint16_t>(vocab_size)) {
                const auto& bytes = vocab[id];
                result.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            } else {
                result += '?';
            }
        }
        return result;
    }
};

#endif // BPE_TOKENIZER_H
