#ifndef MODEL_ZOO_H
#define MODEL_ZOO_H

#include "layer.h"
#include <string>
#include <vector>
#include <functional>
#include <map>

// Model metadata
struct ModelInfo {
    std::string name;
    std::string description;
    std::string dataset;
    std::vector<size_t> input_shape;  // e.g., {1, 28, 28} for MNIST
    size_t num_classes;
    size_t num_params;
    float accuracy;  // Expected accuracy on test set
};

// Model Zoo - registry of pretrained models
class ModelZoo {
public:
    // Get singleton instance
    static ModelZoo& instance();

    // List all available models
    std::vector<std::string> list_models() const;

    // Get model info
    ModelInfo get_info(const std::string& name) const;

    // Check if a model exists
    bool has_model(const std::string& name) const;

    // Check if pretrained weights exist for a model
    bool has_weights(const std::string& name) const;

    // Create model architecture (without pretrained weights)
    Sequential* create(const std::string& name) const;

    // Load pretrained model (architecture + weights)
    // Returns nullptr if weights file not found
    Sequential* load_pretrained(const std::string& name) const;

    // Save a trained model to the zoo
    void save_to_zoo(const std::string& name, Sequential* model);

    // Set the directory where pretrained weights are stored
    void set_weights_dir(const std::string& dir);

    // Get current weights directory
    std::string get_weights_dir() const { return weights_dir_; }

    // Register a custom model (for extensibility)
    using ModelFactory = std::function<Sequential*()>;
    void register_model(const std::string& name, const ModelInfo& info, ModelFactory factory);

private:
    ModelZoo();
    ~ModelZoo() = default;
    ModelZoo(const ModelZoo&) = delete;
    ModelZoo& operator=(const ModelZoo&) = delete;

    void register_builtin_models();

    struct ModelEntry {
        ModelInfo info;
        ModelFactory factory;
    };

    std::map<std::string, ModelEntry> models_;
    std::string weights_dir_;
};

// Convenience functions
inline std::vector<std::string> list_models() {
    return ModelZoo::instance().list_models();
}

inline Sequential* create_model(const std::string& name) {
    return ModelZoo::instance().create(name);
}

inline Sequential* load_pretrained(const std::string& name) {
    return ModelZoo::instance().load_pretrained(name);
}

// Built-in model architecture factories
namespace models {

// MNIST models
Sequential* mnist_mlp();       // Simple MLP: 784 -> 256 -> 128 -> 10
Sequential* mnist_cnn();       // CNN: Conv(16) -> Conv(32) -> FC(128) -> 10

// CIFAR-10 models
Sequential* cifar10_simple();  // Simple CNN: 2 conv blocks + FC
Sequential* cifar10_vgg();     // VGG-style: 3 conv blocks (64->128->256) + FC

// Tiny models for testing/demos
Sequential* tiny_mlp();        // Minimal MLP for quick tests

}  // namespace models

#endif
