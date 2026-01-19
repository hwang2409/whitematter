#include "model_zoo.h"
#include "serialize.h"
#include <cstdio>
#include <sys/stat.h>

ModelZoo& ModelZoo::instance() {
    static ModelZoo zoo;
    return zoo;
}

ModelZoo::ModelZoo() : weights_dir_("pretrained") {
    register_builtin_models();
}

void ModelZoo::register_builtin_models() {
    // MNIST MLP
    register_model("mnist_mlp", {
        "mnist_mlp",
        "Simple MLP for MNIST digit classification",
        "MNIST",
        {1, 784},
        10,
        203530,  // 784*256 + 256 + 256*128 + 128 + 128*10 + 10
        97.5f
    }, models::mnist_mlp);

    // MNIST CNN
    register_model("mnist_cnn", {
        "mnist_cnn",
        "CNN for MNIST digit classification",
        "MNIST",
        {1, 1, 28, 28},
        10,
        207018,
        98.5f
    }, models::mnist_cnn);

    // CIFAR-10 Simple
    register_model("cifar10_simple", {
        "cifar10_simple",
        "Simple CNN for CIFAR-10 classification",
        "CIFAR-10",
        {1, 3, 32, 32},
        10,
        309738,
        75.0f
    }, models::cifar10_simple);

    // CIFAR-10 VGG
    register_model("cifar10_vgg", {
        "cifar10_vgg",
        "VGG-style CNN for CIFAR-10 classification",
        "CIFAR-10",
        {1, 3, 32, 32},
        10,
        3249994,
        85.0f
    }, models::cifar10_vgg);

    // Tiny MLP for testing
    register_model("tiny_mlp", {
        "tiny_mlp",
        "Minimal MLP for quick tests",
        "synthetic",
        {1, 4},
        2,
        42,  // 4*8 + 8 + 8*2 + 2
        100.0f
    }, models::tiny_mlp);
}

void ModelZoo::register_model(const std::string& name, const ModelInfo& info, ModelFactory factory) {
    models_[name] = {info, factory};
}

std::vector<std::string> ModelZoo::list_models() const {
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto& [name, entry] : models_) {
        names.push_back(name);
    }
    return names;
}

ModelInfo ModelZoo::get_info(const std::string& name) const {
    auto it = models_.find(name);
    if (it == models_.end()) {
        return {"", "Model not found", "", {}, 0, 0, 0.0f};
    }
    return it->second.info;
}

bool ModelZoo::has_model(const std::string& name) const {
    return models_.find(name) != models_.end();
}

bool ModelZoo::has_weights(const std::string& name) const {
    std::string path = weights_dir_ + "/" + name + ".bin";
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

Sequential* ModelZoo::create(const std::string& name) const {
    auto it = models_.find(name);
    if (it == models_.end()) {
        fprintf(stderr, "ModelZoo: Unknown model '%s'\n", name.c_str());
        return nullptr;
    }
    return it->second.factory();
}

Sequential* ModelZoo::load_pretrained(const std::string& name) const {
    // First create the architecture
    Sequential* model = create(name);
    if (!model) {
        return nullptr;
    }

    // Try to load weights
    std::string path = weights_dir_ + "/" + name + ".bin";
    if (!has_weights(name)) {
        fprintf(stderr, "ModelZoo: Weights not found at '%s'\n", path.c_str());
        fprintf(stderr, "         Returning model with random initialization.\n");
        fprintf(stderr, "         Train and save with: ModelZoo::instance().save_to_zoo(\"%s\", model)\n", name.c_str());
        return model;
    }

    // Load the weights
    if (!load_model(model, path)) {
        fprintf(stderr, "ModelZoo: Failed to load weights from '%s'\n", path.c_str());
        delete model;
        return nullptr;
    }

    printf("ModelZoo: Loaded pretrained '%s' from %s\n", name.c_str(), path.c_str());
    return model;
}

void ModelZoo::save_to_zoo(const std::string& name, Sequential* model) {
    // Create weights directory if it doesn't exist
    struct stat st;
    if (stat(weights_dir_.c_str(), &st) != 0) {
        #ifdef _WIN32
        mkdir(weights_dir_.c_str());
        #else
        mkdir(weights_dir_.c_str(), 0755);
        #endif
    }

    std::string path = weights_dir_ + "/" + name + ".bin";
    save_model(model, path);
    printf("ModelZoo: Saved '%s' to %s\n", name.c_str(), path.c_str());
}

void ModelZoo::set_weights_dir(const std::string& dir) {
    weights_dir_ = dir;
}

// =============================================================================
// Built-in model architectures
// =============================================================================

namespace models {

Sequential* mnist_mlp() {
    return new Sequential({
        new Flatten(),
        new Linear(784, 256),
        new ReLU(),
        new Linear(256, 128),
        new ReLU(),
        new Linear(128, 10)
    });
}

Sequential* mnist_cnn() {
    return new Sequential({
        // Block 1: 1 -> 16 channels
        new Conv2d(1, 16, 3, 1, 1),
        new BatchNorm2d(16),
        new ReLU(),
        new MaxPool2d(2),  // 28x28 -> 14x14

        // Block 2: 16 -> 32 channels
        new Conv2d(16, 32, 3, 1, 1),
        new BatchNorm2d(32),
        new ReLU(),
        new MaxPool2d(2),  // 14x14 -> 7x7

        // Classifier
        new Flatten(),  // 32*7*7 = 1568
        new Linear(32 * 7 * 7, 128),
        new ReLU(),
        new Linear(128, 10)
    });
}

Sequential* cifar10_simple() {
    return new Sequential({
        // Block 1: 3 -> 32 channels
        new Conv2d(3, 32, 3, 1, 1),
        new BatchNorm2d(32),
        new ReLU(),
        new Conv2d(32, 32, 3, 1, 1),
        new BatchNorm2d(32),
        new ReLU(),
        new MaxPool2d(2),  // 32x32 -> 16x16

        // Block 2: 32 -> 64 channels
        new Conv2d(32, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new Conv2d(64, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new MaxPool2d(2),  // 16x16 -> 8x8

        // Classifier
        new Flatten(),  // 64*8*8 = 4096
        new Linear(64 * 8 * 8, 256),
        new ReLU(),
        new Dropout(0.5),
        new Linear(256, 10)
    });
}

Sequential* cifar10_vgg() {
    return new Sequential({
        // Block 1: 3 -> 64 channels
        new Conv2d(3, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new Conv2d(64, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new MaxPool2d(2),  // 32x32 -> 16x16

        // Block 2: 64 -> 128 channels
        new Conv2d(64, 128, 3, 1, 1),
        new BatchNorm2d(128),
        new ReLU(),
        new Conv2d(128, 128, 3, 1, 1),
        new BatchNorm2d(128),
        new ReLU(),
        new MaxPool2d(2),  // 16x16 -> 8x8

        // Block 3: 128 -> 256 channels
        new Conv2d(128, 256, 3, 1, 1),
        new BatchNorm2d(256),
        new ReLU(),
        new Conv2d(256, 256, 3, 1, 1),
        new BatchNorm2d(256),
        new ReLU(),
        new MaxPool2d(2),  // 8x8 -> 4x4

        // Classifier
        new Flatten(),  // 256*4*4 = 4096
        new Linear(256 * 4 * 4, 512),
        new ReLU(),
        new Dropout(0.5),
        new Linear(512, 10)
    });
}

Sequential* tiny_mlp() {
    return new Sequential({
        new Linear(4, 8),
        new ReLU(),
        new Linear(8, 2)
    });
}

}  // namespace models
