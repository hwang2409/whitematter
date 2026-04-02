/**
 * GPT-Nino Chat: Interactive chat REPL with Nino Nakano.
 *
 * Usage: ./build/gpt_nino_chat <model_path> [temperature]
 *   e.g. ./build/gpt_nino_chat data/nino/nino_gpt.wm 0.4
 *
 * Commands:
 *   quit/exit  — exit the chat
 *   reset      — clear conversation history
 *   temp <val> — change sampling temperature
 */

#include "nino_gpt.h"
#include <sstream>

static const std::string SYSTEM_PROMPT =
    "You are Nino Nakano from The Quintessential Quintuplets. "
    "You are tsundere, proud, initially cold but secretly caring. "
    "You love cooking and are fiercely protective of your sisters.";

static constexpr size_t MAX_GEN_TOKENS = 300;
static const std::string TOKENIZER_PATH = "data/nino/tokenizer.txt";

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [temperature]" << std::endl;
        std::cerr << "  e.g. " << argv[0] << " data/nino/nino_gpt.wm 0.7" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    float temperature = 0.7f;
    if (argc >= 3) {
        temperature = std::stof(argv[2]);
    }

    // ---- Load BPE tokenizer ----
    BPETokenizer tokenizer;
    if (!tokenizer.load(TOKENIZER_PATH)) {
        std::cerr << "ERROR: Cannot load tokenizer from " << TOKENIZER_PATH << std::endl;
        std::cerr << "Run: python examples/preprocess_nino.py" << std::endl;
        return 1;
    }
    std::cout << "Tokenizer loaded: " << tokenizer.vocab_size << " vocab" << std::endl;

    // ---- Load model ----
    std::cout << "Loading model from " << model_path << "..." << std::endl;
    NinoGPT model;
    if (!load_model(&model, model_path)) {
        std::cerr << "ERROR: Failed to load model from " << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded: " << model.count_params() << " parameters" << std::endl;

    std::cout << std::endl;

    // ---- Chat REPL ----
    std::cout << "Nino Nakano Chat (BPE)" << std::endl;
    std::cout << "Type your message (or 'quit' to exit, 'reset' to clear history)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::endl;

    // Initialize conversation with system prompt
    std::string context = "<|system|>" + SYSTEM_PROMPT;

    std::string line;
    while (true) {
        std::cout << "You: ";
        std::cout.flush();

        if (!std::getline(std::cin, line)) break;

        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) line = line.substr(0, end + 1);

        // Handle commands
        if (line == "quit" || line == "exit") {
            std::cout << "Goodbye!" << std::endl;
            break;
        }

        if (line == "reset") {
            context = "<|system|>" + SYSTEM_PROMPT;
            std::cout << "[Conversation reset]" << std::endl;
            std::cout << std::endl;
            continue;
        }

        if (line.substr(0, 5) == "temp ") {
            try {
                temperature = std::stof(line.substr(5));
                std::cout << "[Temperature set to " << temperature << "]" << std::endl;
            } catch (...) {
                std::cout << "[Invalid temperature value]" << std::endl;
            }
            std::cout << std::endl;
            continue;
        }

        // Append user message and Nino tag
        context += "<|user|>" + line + "<|nino|>";

        // Generate Nino's response
        std::string response = generate(model, tokenizer, context, MAX_GEN_TOKENS,
                                         temperature, 0.9f, 1.2f);

        // Trim leading/trailing whitespace from response
        size_t rs = response.find_first_not_of(" \t\r\n");
        if (rs != std::string::npos) response = response.substr(rs);
        size_t re = response.find_last_not_of(" \t\r\n");
        if (re != std::string::npos) response = response.substr(0, re + 1);

        if (response.empty()) {
            response = "...";
        }

        std::cout << "Nino: " << response << std::endl;
        std::cout << std::endl;

        // Update context with Nino's response
        context += response;

        // Truncate context if too long — BPE tokens are ~4 bytes each, so
        // MAX_SEQ_LEN * 4 bytes ≈ MAX_SEQ_LEN BPE tokens of context
        size_t max_context_chars = MAX_SEQ_LEN * 4;
        if (context.size() > max_context_chars * 2) {
            std::string sys = "<|system|>" + SYSTEM_PROMPT;
            std::string recent = context.substr(context.size() - max_context_chars);
            // Find the nearest user/nino tag to avoid cutting mid-message
            size_t tag_pos = recent.find("<|user|>");
            if (tag_pos == std::string::npos) tag_pos = recent.find("<|nino|>");
            if (tag_pos != std::string::npos) {
                recent = recent.substr(tag_pos);
            }
            context = sys + recent;
        }
    }

    return 0;
}
