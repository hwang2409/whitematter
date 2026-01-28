// Standalone inference for custom models
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include "tensor.h"
#include "layer.h"
#include "serialize.h"

// Binary tensor loading (same format as training)
struct TensorFile {
    std::vector<size_t> shape;
    std::vector<float> data;
};

TensorFile load_tensor_file(const std::string& path) {
    TensorFile result;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x54454E53) throw std::runtime_error("Invalid tensor file");

    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);

    result.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim;
        f.read(reinterpret_cast<char*>(&dim), 8);
        result.shape[i] = dim;
    }

    size_t total = 1;
    for (auto d : result.shape) total *= d;
    result.data.resize(total);
    f.read(reinterpret_cast<char*>(result.data.data()), total * sizeof(float));

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.bin> <input_tensor.bin>\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];

    try {
        // Load model
        auto model = load_model(model_path);
        if (!model) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }

        // Load input tensor
        auto input_file = load_tensor_file(input_path);
        auto input = Tensor::create(input_file.shape, false);
        input->data = input_file.data;

        // Run inference
        NoGradGuard no_grad;
        model->eval();
        auto output = model->forward(input);

        // Apply softmax for probabilities
        size_t num_classes = output->shape.back();
        std::vector<float> probs(num_classes);

        float max_val = output->data[0];
        for (size_t i = 1; i < num_classes; i++) {
            max_val = std::max(max_val, output->data[i]);
        }

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; i++) {
            probs[i] = std::exp(output->data[i] - max_val);
            sum += probs[i];
        }
        for (size_t i = 0; i < num_classes; i++) {
            probs[i] /= sum;
        }

        // Find predicted class
        size_t predicted = 0;
        float max_prob = probs[0];
        for (size_t i = 1; i < num_classes; i++) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                predicted = i;
            }
        }

        // Output JSON result
        printf("{\"predicted_class\": %zu, \"probabilities\": [", predicted);
        for (size_t i = 0; i < num_classes; i++) {
            printf("%.6f%s", probs[i], i < num_classes - 1 ? ", " : "");
        }
        printf("]}\n");

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
