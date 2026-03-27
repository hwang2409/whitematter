// ResNet-18 CIFAR-10 inference CLI.
//
// Loads a trained checkpoint and classifies CIFAR-10 test images.
//
// Usage:
//   ./build/resnet18_predict <checkpoint.ckpt> [data_dir]
//
// If data_dir is provided, loads CIFAR-10 test set and runs inference on all
// images, printing accuracy. If not, defaults to "data" directory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "tensor.h"
#include "layer.h"
#include "cifar10.h"
#include "dataloader.h"

// ---------------------------------------------------------------------------
// BasicBlock: the fundamental building block of ResNet-18.
//
//   path A (main):  conv3x3 -> BN -> ReLU -> conv3x3 -> BN
//   path B (skip):  identity (same dims) or conv1x1+BN (downsample)
//   output:         ReLU(A + B)
// ---------------------------------------------------------------------------
struct BasicBlock {
    Conv2d conv1, conv2;
    BatchNorm2d bn1, bn2;

    bool has_downsample;
    Conv2d down_conv;
    BatchNorm2d down_bn;

    BasicBlock(size_t in_ch, size_t out_ch, size_t stride = 1)
        : conv1(in_ch, out_ch, 3, stride, 1),
          conv2(out_ch, out_ch, 3, 1, 1),
          bn1(out_ch),
          bn2(out_ch),
          has_downsample(stride != 1 || in_ch != out_ch),
          down_conv(in_ch, out_ch, 1, stride, 0),
          down_bn(out_ch)
    {}

    TensorPtr forward(const TensorPtr& x) {
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = out->relu();
        out = conv2.forward(out);
        out = bn2.forward(out);

        TensorPtr shortcut;
        if (has_downsample) {
            shortcut = down_conv.forward(x);
            shortcut = down_bn.forward(shortcut);
        } else {
            shortcut = x;
        }

        out = out->add(shortcut);
        out = out->relu();
        return out;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto add = [&](std::vector<TensorPtr> p) {
            params.insert(params.end(), p.begin(), p.end());
        };
        add(conv1.parameters());
        add(bn1.parameters());
        add(conv2.parameters());
        add(bn2.parameters());
        if (has_downsample) {
            add(down_conv.parameters());
            add(down_bn.parameters());
        }
        return params;
    }

    void eval() { bn1.eval(); bn2.eval(); if (has_downsample) down_bn.eval(); }
};

// ---------------------------------------------------------------------------
// ResNet18 for CIFAR-10
// ---------------------------------------------------------------------------
struct ResNet18 {
    Conv2d conv1;
    BatchNorm2d bn1;

    BasicBlock layer1_0, layer1_1;
    BasicBlock layer2_0, layer2_1;
    BasicBlock layer3_0, layer3_1;
    BasicBlock layer4_0, layer4_1;

    AdaptiveAvgPool2d pool;
    Flatten flatten;
    Linear fc;

    ResNet18()
        : conv1(3, 64, 3, 1, 1),
          bn1(64),
          layer1_0(64, 64, 1),
          layer1_1(64, 64, 1),
          layer2_0(64, 128, 2),
          layer2_1(128, 128, 1),
          layer3_0(128, 256, 2),
          layer3_1(256, 256, 1),
          layer4_0(256, 512, 2),
          layer4_1(512, 512, 1),
          pool(1, 1),
          flatten(),
          fc(512, 10)
    {}

    TensorPtr forward(const TensorPtr& x) {
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = out->relu();

        out = layer1_0.forward(out);
        out = layer1_1.forward(out);

        out = layer2_0.forward(out);
        out = layer2_1.forward(out);

        out = layer3_0.forward(out);
        out = layer3_1.forward(out);

        out = layer4_0.forward(out);
        out = layer4_1.forward(out);

        out = pool.forward(out);
        out = flatten.forward(out);
        out = fc.forward(out);
        return out;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto add = [&](std::vector<TensorPtr> p) {
            params.insert(params.end(), p.begin(), p.end());
        };
        add(conv1.parameters());
        add(bn1.parameters());
        add(layer1_0.parameters());
        add(layer1_1.parameters());
        add(layer2_0.parameters());
        add(layer2_1.parameters());
        add(layer3_0.parameters());
        add(layer3_1.parameters());
        add(layer4_0.parameters());
        add(layer4_1.parameters());
        add(fc.parameters());
        return params;
    }

    void eval() {
        bn1.eval();
        layer1_0.eval(); layer1_1.eval();
        layer2_0.eval(); layer2_1.eval();
        layer3_0.eval(); layer3_1.eval();
        layer4_0.eval(); layer4_1.eval();
    }
};

// ---------------------------------------------------------------------------
// Load parameters from checkpoint (weights only, skip optimizer state)
// ---------------------------------------------------------------------------
static const uint32_t CHECKPOINT_MAGIC = 0x574D4350;

static bool load_params(const std::string& path, std::vector<TensorPtr>& params) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != CHECKPOINT_MAGIC) {
        fprintf(stderr, "Invalid checkpoint file: %s (bad magic)\n", path.c_str());
        fclose(f);
        return false;
    }

    int epoch;
    float best_acc, lr;
    fread(&epoch, 4, 1, f);
    fread(&best_acc, 4, 1, f);
    fread(&lr, 4, 1, f);

    uint32_t n;
    fread(&n, 4, 1, f);
    if (n != params.size()) {
        fprintf(stderr, "Parameter count mismatch: checkpoint has %u, model has %zu\n",
                n, params.size());
        fclose(f);
        return false;
    }

    for (auto& p : params) {
        uint32_t ndim;
        fread(&ndim, 4, 1, f);
        for (uint32_t i = 0; i < ndim; i++) {
            uint64_t d;
            fread(&d, 8, 1, f);
        }
        fread(p->data(), sizeof(float), p->size(), f);
    }

    fclose(f);
    printf("Loaded checkpoint: epoch %d, best accuracy %.2f%%\n", epoch, best_acc);
    return true;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <checkpoint.ckpt> [data_dir]\n", argv[0]);
        printf("\n");
        printf("Loads a trained ResNet-18 checkpoint and evaluates on the CIFAR-10 test set.\n");
        printf("\n");
        printf("Arguments:\n");
        printf("  checkpoint.ckpt   Path to a trained checkpoint (e.g. resnet18_best.ckpt)\n");
        printf("  data_dir          Directory containing CIFAR-10 binary files (default: data)\n");
        return 1;
    }

    std::string ckpt_path = argv[1];
    std::string data_dir = argc > 2 ? argv[2] : "data";

    // ------------------------------------------------------------------
    // Build model and load checkpoint
    // ------------------------------------------------------------------
    ResNet18 model;
    auto params = model.parameters();

    printf("ResNet-18 CIFAR-10 Inference\n");
    printf("============================\n\n");

    size_t total_params = 0;
    for (const auto& p : params) {
        total_params += p->size();
    }
    printf("Model parameters: %zu (~%.2f MB)\n", total_params,
           total_params * 4.0f / (1024.0f * 1024.0f));

    printf("Loading checkpoint: %s\n", ckpt_path.c_str());
    if (!load_params(ckpt_path, params)) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", ckpt_path.c_str());
        return 1;
    }

    model.eval();
    printf("\nModel set to eval mode (BatchNorm using running statistics)\n\n");

    // ------------------------------------------------------------------
    // Load CIFAR-10 test set
    // ------------------------------------------------------------------
    printf("Loading CIFAR-10 test set from '%s'...\n", data_dir.c_str());
    CIFAR10Dataset test_data;
    try {
        test_data = load_cifar10_test(data_dir);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading CIFAR-10: %s\n", e.what());
        fprintf(stderr, "\nExpected files in %s: test_batch.bin\n", data_dir.c_str());
        fprintf(stderr, "Download: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n");
        return 1;
    }
    printf("Test samples: %zu\n\n", test_data.num_samples);

    // ------------------------------------------------------------------
    // Run inference (no gradient tracking)
    // ------------------------------------------------------------------
    const char* class_names[] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    NoGradGuard no_grad;

    size_t correct = 0;
    size_t total = 0;
    size_t batch_size = 100;

    Dataset test_dataset{test_data.images, test_data.labels, test_data.num_samples};
    ThreadedDataLoader loader(test_dataset, batch_size, false, 0);

    printf("Running inference (batch_size=%zu)...\n\n", batch_size);
    printf("First 20 predictions:\n");

    while (loader.has_next()) {
        auto [images, labels] = loader.next_batch_pair();
        auto output = model.forward(images);

        size_t bs = output->shape[0];
        size_t nc = output->shape[1];
        for (size_t i = 0; i < bs; i++) {
            size_t pred = 0;
            float max_val = output->data()[i * nc];
            for (size_t j = 1; j < nc; j++) {
                if (output->data()[i * nc + j] > max_val) {
                    max_val = output->data()[i * nc + j];
                    pred = j;
                }
            }
            size_t label = static_cast<size_t>(labels->data()[i]);
            if (pred == label) correct++;
            total++;

            if (total <= 20) {
                printf("  Image %3zu: predicted=%-12s (%.1f%%), actual=%-12s %s\n",
                       total, class_names[pred], max_val * 100.0f,
                       class_names[label], pred == label ? "[OK]" : "[X]");
            }
        }

        // Progress indicator every 10 batches
        if (total > 20 && (total / batch_size) % 10 == 0) {
            printf("\r  Processed %zu/%zu images...", total, test_data.num_samples);
            fflush(stdout);
        }
    }

    printf("\n\nTest accuracy: %zu/%zu = %.2f%%\n", correct, total,
           100.0f * static_cast<float>(correct) / static_cast<float>(total));

    return 0;
}
