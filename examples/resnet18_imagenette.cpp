// ResNet-18 for ImageNette (224x224, 10 classes) with CUDA GPU acceleration.
//
// This is the standard ImageNet ResNet-18 architecture (7x7 initial conv,
// maxpool, 4 residual layers), trained on ImageNette — a 10-class subset
// of ImageNet.  All CUDA-specific code is behind #ifdef WHITEMATTER_CUDA
// so the file compiles and runs on CPU when CUDA is not available.
//
// Data is loaded from binary tensor files (same format as cats_vs_dogs):
//   magic (4B = 0x54454E53) | ndim (4B) | shape (ndim * 8B) | float data
//
// Usage:
//   ./build/resnet18_imagenette data/imagenette [batch_size] [--resume checkpoint.ckpt]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <sys/stat.h>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "dataloader.h"
#include "serialize.h"
#include "device.h"
#include "memory_pool.h"

#ifdef WHITEMATTER_CUDA
#include "cuda/cuda_backend.h"
#endif

// ---------------------------------------------------------------------------
// ImageNette class names (10-class subset of ImageNet)
// ---------------------------------------------------------------------------
static const char* class_names[] = {
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
};

// ---------------------------------------------------------------------------
// Binary tensor file loader (same format as cats_vs_dogs)
// ---------------------------------------------------------------------------
static TensorPtr load_tensor_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic, ndim;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x54454E53) throw std::runtime_error("Bad magic in: " + path);

    f.read(reinterpret_cast<char*>(&ndim), 4);

    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim;
        f.read(reinterpret_cast<char*>(&dim), 8);
        shape[i] = static_cast<size_t>(dim);
    }

    auto tensor = Tensor::create(shape, false);
    size_t num_floats = tensor->size();
    f.read(reinterpret_cast<char*>(tensor->data()), num_floats * sizeof(float));

    if (!f) throw std::runtime_error("Truncated data in: " + path);
    return tensor;
}

// ---------------------------------------------------------------------------
// Checkpoint format (binary):
//   [magic:u32 = 0x574D4350]  "WMCP"
//   [epoch:i32]               epoch after which this checkpoint was saved
//   [best_acc:f32]            best test accuracy seen so far
//   [lr:f32]                  current learning rate
//   [num_params:u32]
//   for each parameter:
//     [ndim:u32][dims:u64...][data:float...]
//   [num_momentum:u32]
//   for each momentum buffer:
//     [size:u64][data:float...]
//   [num_bn_stats:u32]
//   for each BN running stat:
//     [size:u64][data:float...]
// ---------------------------------------------------------------------------
static const uint32_t CHECKPOINT_MAGIC = 0x574D4350;

static void save_checkpoint(const std::string& path, int epoch, float best_acc,
                            float lr, const std::vector<TensorPtr>& params,
                            const std::vector<std::vector<float>>& momentum,
                            const std::vector<TensorPtr>& bn_stats = {}) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Failed to save checkpoint to %s\n", path.c_str()); return; }

    fwrite(&CHECKPOINT_MAGIC, 4, 1, f);
    fwrite(&epoch, 4, 1, f);
    fwrite(&best_acc, 4, 1, f);
    fwrite(&lr, 4, 1, f);

    // Save parameters
    uint32_t n = params.size();
    fwrite(&n, 4, 1, f);
    for (auto& p : params) {
        uint32_t ndim = p->shape.size();
        fwrite(&ndim, sizeof(uint32_t), 1, f);
        for (auto s : p->shape) {
            uint64_t dim = s;
            fwrite(&dim, sizeof(uint64_t), 1, f);
        }
        fwrite(p->data(), sizeof(float), p->size(), f);
    }

    // Save momentum buffers
    uint32_t nm = momentum.size();
    fwrite(&nm, 4, 1, f);
    for (auto& buf : momentum) {
        uint64_t sz = buf.size();
        fwrite(&sz, 8, 1, f);
        fwrite(buf.data(), sizeof(float), buf.size(), f);
    }

    // Save BatchNorm running stats (running_mean, running_var for each BN layer)
    uint32_t nbn = bn_stats.size();
    fwrite(&nbn, 4, 1, f);
    for (auto& s : bn_stats) {
        uint64_t sz = s->size();
        fwrite(&sz, 8, 1, f);
        fwrite(s->data(), sizeof(float), s->size(), f);
    }

    fclose(f);
    printf("Checkpoint saved to %s (epoch %d, best %.2f%%, lr %.6f)\n",
           path.c_str(), epoch, best_acc, lr);
}

static bool load_checkpoint(const std::string& path, int& epoch, float& best_acc,
                            float& lr, std::vector<TensorPtr>& params,
                            std::vector<std::vector<float>>& momentum) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != CHECKPOINT_MAGIC) {
        fprintf(stderr, "Invalid checkpoint file: %s (bad magic)\n", path.c_str());
        fclose(f);
        return false;
    }

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

    // Load momentum buffers
    uint32_t nm;
    if (fread(&nm, 4, 1, f) == 1 && nm == momentum.size()) {
        for (auto& buf : momentum) {
            uint64_t sz;
            fread(&sz, 8, 1, f);
            if (sz == buf.size()) {
                fread(buf.data(), sizeof(float), buf.size(), f);
            } else {
                fseek(f, sz * sizeof(float), SEEK_CUR);
                std::fill(buf.begin(), buf.end(), 0.0f);
            }
        }
    }

    fclose(f);
    return true;
}

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

    // Downsample path (only when stride != 1 or channels change)
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
        // Main path
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = out->relu();
        out = conv2.forward(out);
        out = bn2.forward(out);

        // Skip connection
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

    std::vector<TensorPtr> running_stats() {
        std::vector<TensorPtr> stats;
        stats.push_back(bn1.running_mean); stats.push_back(bn1.running_var);
        stats.push_back(bn2.running_mean); stats.push_back(bn2.running_var);
        if (has_downsample) {
            stats.push_back(down_bn.running_mean); stats.push_back(down_bn.running_var);
        }
        return stats;
    }

    void train() { bn1.train(); bn2.train(); if (has_downsample) down_bn.train(); }
    void eval()  { bn1.eval();  bn2.eval();  if (has_downsample) down_bn.eval();  }

    void to(whitematter::DeviceType d) {
        for (auto& p : conv1.parameters()) p->to_inplace(d);
        for (auto& p : bn1.parameters())   p->to_inplace(d);
        for (auto& p : conv2.parameters()) p->to_inplace(d);
        for (auto& p : bn2.parameters())   p->to_inplace(d);
        if (has_downsample) {
            for (auto& p : down_conv.parameters()) p->to_inplace(d);
            for (auto& p : down_bn.parameters())   p->to_inplace(d);
        }
    }
};

// ---------------------------------------------------------------------------
// ResNet-18 for ImageNette (standard ImageNet variant, 224x224 input)
//
// Key differences from CIFAR variant:
//   - First conv: 7x7, stride 2, padding 3 (not 3x3, stride 1, padding 1)
//   - MaxPool(3, stride=2) after first conv+BN+ReLU
//   - Resolution path: 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1
// ---------------------------------------------------------------------------
struct ResNet18 {
    // Initial conv + BN (7x7, stride 2, padding 3)
    Conv2d conv1;
    BatchNorm2d bn1;

    // 4 layers, 2 blocks each
    BasicBlock layer1_0, layer1_1;   // 64  channels, 56x56
    BasicBlock layer2_0, layer2_1;   // 128 channels, 28x28
    BasicBlock layer3_0, layer3_1;   // 256 channels, 14x14
    BasicBlock layer4_0, layer4_1;   // 512 channels, 7x7

    // Classifier head
    AdaptiveAvgPool2d pool;
    Flatten flatten;
    Linear fc;

    ResNet18()
        : conv1(3, 64, 7, 2, 3),          // 224x224 -> 112x112
          bn1(64),
          layer1_0(64, 64, 1),             // 56x56
          layer1_1(64, 64, 1),             // 56x56
          layer2_0(64, 128, 2),            // 56x56 -> 28x28
          layer2_1(128, 128, 1),           // 28x28
          layer3_0(128, 256, 2),           // 28x28 -> 14x14
          layer3_1(256, 256, 1),           // 14x14
          layer4_0(256, 512, 2),           // 14x14 -> 7x7
          layer4_1(512, 512, 1),           // 7x7
          pool(1, 1),
          flatten(),
          fc(512, 10)
    {}

    TensorPtr forward(const TensorPtr& x) {
        // Stem: conv7x7 -> BN -> ReLU -> MaxPool
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = out->relu();
        // MaxPool(3, stride=2, padding=1): pad first, then maxpool
        out = out->pad2d(1)->maxpool2d(3, 2);  // 112x112 -> 56x56

        // Residual layers
        out = layer1_0.forward(out);
        out = layer1_1.forward(out);

        out = layer2_0.forward(out);
        out = layer2_1.forward(out);

        out = layer3_0.forward(out);
        out = layer3_1.forward(out);

        out = layer4_0.forward(out);
        out = layer4_1.forward(out);

        // Classifier
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

    std::vector<TensorPtr> running_stats() {
        std::vector<TensorPtr> stats;
        auto add = [&](std::vector<TensorPtr> s) {
            stats.insert(stats.end(), s.begin(), s.end());
        };
        stats.push_back(bn1.running_mean); stats.push_back(bn1.running_var);
        add(layer1_0.running_stats()); add(layer1_1.running_stats());
        add(layer2_0.running_stats()); add(layer2_1.running_stats());
        add(layer3_0.running_stats()); add(layer3_1.running_stats());
        add(layer4_0.running_stats()); add(layer4_1.running_stats());
        return stats;
    }

    void train() {
        bn1.train();
        layer1_0.train(); layer1_1.train();
        layer2_0.train(); layer2_1.train();
        layer3_0.train(); layer3_1.train();
        layer4_0.train(); layer4_1.train();
    }

    void eval() {
        bn1.eval();
        layer1_0.eval(); layer1_1.eval();
        layer2_0.eval(); layer2_1.eval();
        layer3_0.eval(); layer3_1.eval();
        layer4_0.eval(); layer4_1.eval();
    }

    void to(whitematter::DeviceType d) {
        for (auto& p : conv1.parameters()) p->to_inplace(d);
        for (auto& p : bn1.parameters())   p->to_inplace(d);
        layer1_0.to(d); layer1_1.to(d);
        layer2_0.to(d); layer2_1.to(d);
        layer3_0.to(d); layer3_1.to(d);
        layer4_0.to(d); layer4_1.to(d);
        for (auto& p : fc.parameters()) p->to_inplace(d);
    }
};

// ---------------------------------------------------------------------------
// Helper: transfer output tensor to CPU for accuracy calculation
// ---------------------------------------------------------------------------
static TensorPtr ensure_cpu(const TensorPtr& t) {
#ifdef WHITEMATTER_CUDA
    if (t->is_cuda()) {
        return t->to(whitematter::DeviceType::CPU);
    }
#endif
    return t;
}

// ---------------------------------------------------------------------------
// Helper: read scalar loss value (may be on GPU)
// ---------------------------------------------------------------------------
static float read_loss_scalar(const TensorPtr& loss) {
#ifdef WHITEMATTER_CUDA
    if (loss->is_cuda()) {
        float val = 0.0f;
        whitematter::CUDABackend::instance().memcpy_d2h(&val, loss->data(), 1);
        return val;
    }
#endif
    return loss->data()[0];
}

// ---------------------------------------------------------------------------
// Evaluation helper
// ---------------------------------------------------------------------------
static float compute_accuracy(ResNet18& model, ThreadedDataLoader& loader,
                              bool /*use_cuda*/) {
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0;
    size_t total = 0;

    loader.reset();
    while (loader.has_next()) {
        auto [images, labels] = loader.next_batch_pair();

        auto output = model.forward(images);

        auto cpu_output = ensure_cpu(output);
        auto cpu_labels = ensure_cpu(labels);

        size_t batch_size = cpu_output->shape[0];
        size_t num_classes = cpu_output->shape[1];

        for (size_t i = 0; i < batch_size; i++) {
            size_t predicted = 0;
            float max_val = cpu_output->data()[i * num_classes];
            for (size_t j = 1; j < num_classes; j++) {
                if (cpu_output->data()[i * num_classes + j] > max_val) {
                    max_val = cpu_output->data()[i * num_classes + j];
                    predicted = j;
                }
            }
            if (predicted == static_cast<size_t>(cpu_labels->data()[i])) {
                correct++;
            }
            total++;
        }
    }

    model.train();
    return static_cast<float>(correct) / static_cast<float>(total) * 100.0f;
}

// ---------------------------------------------------------------------------
// Manual L2 weight decay: add weight_decay * w to each gradient before the
// optimizer step.  This is equivalent to SGD with weight_decay in PyTorch.
// We only decay conv and linear weights, NOT batchnorm gamma/beta.
// ---------------------------------------------------------------------------
static void apply_weight_decay(std::vector<TensorPtr>& params, float wd) {
    for (auto& p : params) {
        // Skip 1-D parameters (biases, BN gamma/beta)
        if (p->shape.size() <= 1) continue;
        if (!p->grad()) continue;

        for (size_t j = 0; j < p->size(); j++) {
            p->grad()[j] += wd * p->data()[j];
        }
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = "data/imagenette";
    size_t batch_size = 32;
    std::string resume_path;

    // Parse arguments: [data_dir] [batch_size] [--resume checkpoint.ckpt]
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--resume" && i + 1 < argc) {
            resume_path = argv[++i];
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() >= 1) data_dir   = positional[0];
    if (positional.size() >= 2) batch_size = std::atoi(positional[1].c_str());

    // ------------------------------------------------------------------
    // Check CUDA availability
    // ------------------------------------------------------------------
    bool use_cuda = false;
#ifdef WHITEMATTER_CUDA
    if (whitematter::cuda_backend_available()) {
        whitematter::CUDABackend::instance().init();
        use_cuda = true;
    }
#endif

    printf("ResNet-18 ImageNette Training (%s)\n",
           use_cuda ? "CUDA GPU accelerated" : "CPU");
    printf("====================================\n\n");

    if (use_cuda) {
        printf("Device: CUDA GPU (model + data on GPU, backward via CPU bridge)\n\n");
    } else {
        printf("Device: CPU\n\n");
    }
    fflush(stdout);

    // ------------------------------------------------------------------
    // Load dataset from binary tensor files
    // ------------------------------------------------------------------
    printf("Loading ImageNette dataset from '%s'...\n", data_dir.c_str());
    fflush(stdout);

    TensorPtr train_images, train_labels, test_images, test_labels;
    try {
        train_images = load_tensor_bin(data_dir + "/train_images.bin");
        train_labels = load_tensor_bin(data_dir + "/train_labels.bin");
        test_images  = load_tensor_bin(data_dir + "/test_images.bin");
        test_labels  = load_tensor_bin(data_dir + "/test_labels.bin");
    } catch (const std::exception& e) {
        printf("Error loading data: %s\n", e.what());
        printf("\nRun the preprocessing script first:\n");
        printf("  python examples/preprocess_imagenette.py\n");
        printf("\nExpected files in '%s':\n", data_dir.c_str());
        printf("  train_images.bin  (N, 3, 224, 224) float32\n");
        printf("  train_labels.bin  (N,) float32\n");
        printf("  test_images.bin   (N, 3, 224, 224) float32\n");
        printf("  test_labels.bin   (N,) float32\n");
        return 1;
    }

    size_t n_train = train_images->shape[0];
    size_t n_test  = test_images->shape[0];
    printf("Train: %zu samples | Test: %zu samples\n", n_train, n_test);
    printf("Image shape: [%zu, %zu, %zu]\n\n",
           train_images->shape[1], train_images->shape[2], train_images->shape[3]);

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------
    const int    num_epochs   = 90;
    const float  init_lr      = 0.1f;
    const float  weight_decay = 1e-4f;
    const float  momentum_val = 0.9f;

    // ------------------------------------------------------------------
    // Model
    // ------------------------------------------------------------------
    ResNet18 model;
    auto all_params = model.parameters();

    size_t total_params = 0;
    for (const auto& p : all_params) {
        total_params += p->size();
    }

    printf("Architecture: ResNet-18 (ImageNet variant, 224x224 input)\n");
    printf("  Conv(3,64,7,s2,p3) -> BN -> ReLU -> MaxPool(3,s2,p1)\n");
    printf("  Layer1: BasicBlock(64,64)  x2   [56x56]\n");
    printf("  Layer2: BasicBlock(64,128,s2) + BasicBlock(128,128)  [28x28]\n");
    printf("  Layer3: BasicBlock(128,256,s2) + BasicBlock(256,256) [14x14]\n");
    printf("  Layer4: BasicBlock(256,512,s2) + BasicBlock(512,512) [7x7]\n");
    printf("  AdaptiveAvgPool(1,1) -> Flatten -> Linear(512,10)\n");
    printf("  Total parameters: %zu (~%.2f MB)\n\n", total_params,
           total_params * 4.0f / (1024.0f * 1024.0f));

    printf("Optimizer: SGD (lr=%.2f, momentum=%.1f, weight_decay=%.4f)\n",
           init_lr, momentum_val, weight_decay);
    printf("Scheduler: CosineAnnealingLR (T_max=%d, eta_min=0)\n", num_epochs);
    printf("Batch size: %zu\n", batch_size);
    printf("Epochs: %d\n", num_epochs);
    printf("Data augmentation: pad(16) -> random_crop(224,224) -> random_flip_horizontal(0.5)\n\n");
    fflush(stdout);

    if (use_cuda) {
        printf("GPU acceleration: matmul offloaded to cuBLAS for large matrices.\n\n");
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Optimizer & scheduler
    // ------------------------------------------------------------------
    SGD optimizer(all_params, init_lr, momentum_val);
    CosineAnnealingLR scheduler(&optimizer, num_epochs, 0.0f);
    CrossEntropyLoss criterion;

    // ------------------------------------------------------------------
    // Data loaders
    // ------------------------------------------------------------------
    Dataset train_dataset{train_images, train_labels, n_train};
    Dataset test_dataset{test_images, test_labels, n_test};

    ThreadedDataLoader train_loader(train_dataset, batch_size, true, 2);
    ThreadedDataLoader test_loader(test_dataset, batch_size, false, 0);

    // ------------------------------------------------------------------
    // Resume from checkpoint (if --resume was specified)
    // ------------------------------------------------------------------
    int start_epoch = 0;
    float best_test_acc = 0.0f;

    if (!resume_path.empty()) {
        int saved_epoch = 0;
        float saved_acc = 0.0f, saved_lr = 0.0f;
        if (load_checkpoint(resume_path, saved_epoch, saved_acc, saved_lr,
                            all_params, optimizer.velocity)) {
            start_epoch = saved_epoch;
            best_test_acc = saved_acc;
            optimizer.lr = saved_lr;
            scheduler.last_epoch = start_epoch - 1;
            scheduler.step();
            printf("Resumed from checkpoint: %s\n", resume_path.c_str());
            printf("  Continuing from epoch %d, best acc %.2f%%, lr %.6f\n\n",
                   start_epoch, best_test_acc, optimizer.lr);
        } else {
            printf("Warning: could not load checkpoint '%s', training from scratch.\n\n",
                   resume_path.c_str());
        }
    }

    // ------------------------------------------------------------------
    // Create checkpoint directory
    // ------------------------------------------------------------------
    mkdir("checkpoints", 0755);

    // ------------------------------------------------------------------
    // Training loop
    // ------------------------------------------------------------------
    printf("Training...\n");
    printf("--------------------------------------------------------------------------------\n");
    fflush(stdout);

    for (int epoch = start_epoch; epoch < num_epochs; epoch++) {
        model.train();
        train_loader.reset();

        auto epoch_start = std::chrono::high_resolution_clock::now();

        float total_loss  = 0.0f;
        size_t correct    = 0;
        size_t total      = 0;
        size_t num_batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch_pair();

            // Data augmentation: pad 16 -> random crop 224x224 -> random horizontal flip
            auto augmented = images->pad2d(16)->random_crop(224, 224)->random_flip_horizontal(0.5f);

            optimizer.zero_grad();

            auto output = model.forward(augmented);
            auto loss   = criterion(output, labels);

            loss->backward();

            // Manual L2 weight decay on conv/linear weights
            apply_weight_decay(all_params, weight_decay);

            optimizer.step();

            // Free cached memory to prevent unbounded growth with large 224x224 tensors
            MemoryPool::instance().trim();

            // Read loss scalar (handles GPU -> CPU transfer)
            total_loss += read_loss_scalar(loss);
            num_batches++;

            // Compute training accuracy for this batch
            auto cpu_output = ensure_cpu(output);
            auto cpu_labels = ensure_cpu(labels);
            size_t bs = cpu_output->shape[0];
            size_t nc = cpu_output->shape[1];
            for (size_t i = 0; i < bs; i++) {
                size_t pred = 0;
                float mv = cpu_output->data()[i * nc];
                for (size_t j = 1; j < nc; j++) {
                    if (cpu_output->data()[i * nc + j] > mv) {
                        mv = cpu_output->data()[i * nc + j];
                        pred = j;
                    }
                }
                if (pred == static_cast<size_t>(cpu_labels->data()[i])) correct++;
                total++;
            }

            // Print progress
            if (epoch == start_epoch || num_batches % 20 == 0) {
                auto batch_time = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(batch_time - epoch_start).count();
                printf("\r  Epoch %3d: batch %3zu/%zu | %.1fs elapsed",
                       epoch + 1, num_batches, train_loader.num_batches(), elapsed);
                fflush(stdout);
            }
        }

        scheduler.step();

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_secs = std::chrono::duration<double>(epoch_end - epoch_start).count();

        float avg_loss  = total_loss / static_cast<float>(num_batches);
        float train_acc = static_cast<float>(correct) / static_cast<float>(total) * 100.0f;
        float test_acc  = compute_accuracy(model, test_loader, use_cuda);

        bool saved_best = false;
        if (test_acc > best_test_acc) {
            best_test_acc = test_acc;
            save_checkpoint("checkpoints/resnet18_imagenette_best.ckpt", epoch + 1, best_test_acc,
                            optimizer.lr, all_params, optimizer.velocity, model.running_stats());
            saved_best = true;
        }
        if ((epoch + 1) % 10 == 0) {
            save_checkpoint("checkpoints/resnet18_imagenette_latest.ckpt", epoch + 1, best_test_acc,
                            optimizer.lr, all_params, optimizer.velocity, model.running_stats());
        }

        printf("\r  Epoch %3d | Loss: %.4f | Train: %.2f%% | Test: %.2f%% | Best: %.2f%% | LR: %.6f | %.1fs",
               epoch + 1, avg_loss, train_acc, test_acc, best_test_acc, optimizer.lr, epoch_secs);
        if (saved_best) printf(" [saved best]");
        printf("\n");
        fflush(stdout);
    }

    printf("--------------------------------------------------------------------------------\n");
    save_checkpoint("checkpoints/resnet18_imagenette_final.ckpt", num_epochs, best_test_acc,
                    optimizer.lr, all_params, optimizer.velocity, model.running_stats());
    printf("Training complete! Best test accuracy: %.2f%%\n\n", best_test_acc);

    // ------------------------------------------------------------------
    // Sample predictions
    // ------------------------------------------------------------------
    printf("Sample predictions:\n");
    {
        NoGradGuard no_grad;
        model.eval();
        test_loader.reset();
        auto [images, labels] = test_loader.next_batch_pair();

        auto output = model.forward(images);

        auto cpu_output = ensure_cpu(output);
        auto cpu_labels = ensure_cpu(labels);

        int num_samples = std::min(10, static_cast<int>(cpu_output->shape[0]));
        for (int i = 0; i < num_samples; i++) {
            size_t predicted = 0;
            float max_val = cpu_output->data()[i * 10];
            for (size_t j = 1; j < 10; j++) {
                if (cpu_output->data()[i * 10 + j] > max_val) {
                    max_val = cpu_output->data()[i * 10 + j];
                    predicted = j;
                }
            }
            int actual = static_cast<int>(cpu_labels->data()[i]);
            const char* status = (predicted == static_cast<size_t>(actual)) ? "[OK]" : "[X]";
            printf("  %s Predicted: %-18s  Actual: %-18s\n",
                   status, class_names[predicted], class_names[actual]);
        }
    }

    return 0;
}
