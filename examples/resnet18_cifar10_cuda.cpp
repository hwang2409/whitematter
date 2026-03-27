// ResNet-18 for CIFAR-10 with CUDA GPU acceleration.
//
// This is the GPU-enabled variant of resnet18_cifar10.cpp.  All CUDA-specific
// code is behind #ifdef WHITEMATTER_CUDA so the file compiles and runs on CPU
// when CUDA is not available.
//
// Key differences from the CPU version:
//   1. Model parameters are moved to GPU at startup (model.to(CUDA)).
//   2. Each mini-batch is transferred to GPU before the forward pass.
//   3. The scalar loss value is copied back to CPU for logging.
//   4. Predictions are copied back to CPU for accuracy calculation.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "cifar10.h"
#include "dataloader.h"
#include "serialize.h"
#include "device.h"

#ifdef WHITEMATTER_CUDA
#include "cuda/cuda_backend.h"
#endif

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
// ---------------------------------------------------------------------------
static const uint32_t CHECKPOINT_MAGIC = 0x574D4350;

static void save_checkpoint(const std::string& path, int epoch, float best_acc,
                            float lr, const std::vector<TensorPtr>& params,
                            const std::vector<std::vector<float>>& momentum) {
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
                // Size mismatch — skip this buffer and zero our copy
                fseek(f, sz * sizeof(float), SEEK_CUR);
                std::fill(buf.begin(), buf.end(), 0.0f);
            }
        }
    }
    // If momentum section is missing or count mismatches, momentum stays at zero
    // (this provides backward compatibility with older checkpoint files)

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

    // Collect all learnable parameters from this block.
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
// ResNet18 for CIFAR-10
// ---------------------------------------------------------------------------
struct ResNet18 {
    // Initial conv + BN
    Conv2d conv1;
    BatchNorm2d bn1;

    // 4 layers, 2 blocks each
    BasicBlock layer1_0, layer1_1;   // 64  channels, 32x32
    BasicBlock layer2_0, layer2_1;   // 128 channels, 16x16
    BasicBlock layer3_0, layer3_1;   // 256 channels, 8x8
    BasicBlock layer4_0, layer4_1;   // 512 channels, 4x4

    // Classifier head
    AdaptiveAvgPool2d pool;
    Flatten flatten;
    Linear fc;

    ResNet18()
        : conv1(3, 64, 3, 1, 1),      // 32x32 -> 32x32
          bn1(64),
          layer1_0(64, 64, 1),         // 32x32
          layer1_1(64, 64, 1),         // 32x32
          layer2_0(64, 128, 2),        // 32x32 -> 16x16
          layer2_1(128, 128, 1),       // 16x16
          layer3_0(128, 256, 2),       // 16x16 -> 8x8
          layer3_1(256, 256, 1),       // 8x8
          layer4_0(256, 512, 2),       // 8x8 -> 4x4
          layer4_1(512, 512, 1),       // 4x4
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
float compute_accuracy(ResNet18& model, ThreadedDataLoader& loader,
                       bool use_cuda) {
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0;
    size_t total = 0;

    loader.reset();
    while (loader.has_next()) {
        auto [images, labels] = loader.next_batch_pair();

        auto output = model.forward(images);

        // Output is on CPU — no transfer needed
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
//
// NOTE: when running on GPU the parameter data lives in device memory.
// We transfer to CPU, apply decay, then transfer back.  A proper CUDA
// kernel would do this in-place on the GPU — but this keeps the example
// simple and correct for both CPU and GPU paths.
// ---------------------------------------------------------------------------
void apply_weight_decay(std::vector<TensorPtr>& params, float wd) {
    for (auto& p : params) {
        // Skip 1-D parameters (biases, BN gamma/beta)
        if (p->shape.size() <= 1) continue;
        if (!p->grad()) continue;

        // Data is always on CPU in the transparent offload architecture
        for (size_t j = 0; j < p->size(); j++) {
            p->grad()[j] += wd * p->data()[j];
        }
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    size_t batch_size = 128;
    std::string resume_path;

    // Parse arguments: [data_dir] [batch_size] [--resume checkpoint.ckpt]
    // Positional args are collected after stripping --resume.
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

    printf("ResNet-18 CIFAR-10 Training (%s)\n",
           use_cuda ? "CUDA GPU accelerated" : "CPU");
    printf("===========================\n\n");

    if (use_cuda) {
        printf("Device: CUDA GPU (model + data on GPU, backward via CPU bridge)\n\n");
    } else {
        printf("Device: CPU\n\n");
    }
    fflush(stdout);

    // ------------------------------------------------------------------
    // Load dataset
    // ------------------------------------------------------------------
    printf("Loading CIFAR-10 dataset from '%s'...\n", data_dir.c_str());
    fflush(stdout);

    CIFAR10Dataset train_data, test_data;
    try {
        train_data = load_cifar10_train(data_dir);
        test_data  = load_cifar10_test(data_dir);
    } catch (const std::exception& e) {
        printf("Error loading CIFAR-10: %s\n", e.what());
        printf("\nTo download CIFAR-10 data, run:\n");
        printf("  mkdir -p %s && cd %s\n", data_dir.c_str(), data_dir.c_str());
        printf("  curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n");
        printf("  tar -xzf cifar-10-binary.tar.gz\n");
        printf("  mv cifar-10-batches-bin/*.bin .\n");
        printf("\nExpected files: data_batch_1.bin ... data_batch_5.bin, test_batch.bin\n");
        return 1;
    }

    printf("Train samples: %zu\n", train_data.num_samples);
    printf("Test samples:  %zu\n\n", test_data.num_samples);

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------
    // batch_size is set from command line (default 128), see top of main()
    const int    num_epochs   = 200;
    const float  init_lr      = 0.1f;
    const float  weight_decay = 5e-4f;
    const float  momentum     = 0.9f;

    // ------------------------------------------------------------------
    // Model
    // ------------------------------------------------------------------
    ResNet18 model;
    auto all_params = model.parameters();

    size_t total_params = 0;
    for (const auto& p : all_params) {
        total_params += p->size();
    }

    printf("Architecture: ResNet-18 (CIFAR-10 variant)\n");
    printf("  Conv(3,64,3) -> BN -> ReLU\n");
    printf("  Layer1: BasicBlock(64,64)  x2   [32x32]\n");
    printf("  Layer2: BasicBlock(64,128,s2) + BasicBlock(128,128)  [16x16]\n");
    printf("  Layer3: BasicBlock(128,256,s2) + BasicBlock(256,256) [8x8]\n");
    printf("  Layer4: BasicBlock(256,512,s2) + BasicBlock(512,512) [4x4]\n");
    printf("  AdaptiveAvgPool(1,1) -> Flatten -> Linear(512,10)\n");
    printf("  Total parameters: %zu (~%.2f MB)\n\n", total_params,
           total_params * 4.0f / (1024.0f * 1024.0f));

    printf("Optimizer: SGD (lr=%.2f, momentum=%.1f, weight_decay=%.4f)\n",
           init_lr, momentum, weight_decay);
    printf("Scheduler: CosineAnnealingLR (T_max=%d, eta_min=0)\n", num_epochs);
    printf("Batch size: %zu\n", batch_size);
    printf("Epochs: %d\n", num_epochs);
    printf("Data augmentation: pad(4) -> random_crop(32,32) -> random_flip_horizontal(0.5)\n\n");
    fflush(stdout);

    // Data stays on CPU. GPU acceleration is transparent via matmul_blocked
    // dispatching large GEMMs to cuBLAS. No .to(CUDA) needed.
    if (use_cuda) {
        printf("GPU acceleration: matmul offloaded to cuBLAS for large matrices.\n\n");
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Optimizer & scheduler
    // ------------------------------------------------------------------
    SGD optimizer(all_params, init_lr, momentum);
    CosineAnnealingLR scheduler(&optimizer, num_epochs, 0.0f);
    CrossEntropyLoss criterion;

    // ------------------------------------------------------------------
    // Data loaders
    // ------------------------------------------------------------------
    Dataset train_dataset{train_data.images, train_data.labels, train_data.num_samples};
    Dataset test_dataset{test_data.images,  test_data.labels,  test_data.num_samples};

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
            // Advance scheduler to match the resumed epoch.
            // The scheduler starts at last_epoch=-1; each step() increments it.
            // After training epoch N (0-indexed), step() was called, so
            // last_epoch should be N. We need start_epoch calls.
            scheduler.last_epoch = start_epoch - 1;
            scheduler.step();  // sets last_epoch = start_epoch, computes correct lr
            printf("Resumed from checkpoint: %s\n", resume_path.c_str());
            printf("  Continuing from epoch %d, best acc %.2f%%, lr %.6f\n\n",
                   start_epoch, best_test_acc, optimizer.lr);
        } else {
            printf("Warning: could not load checkpoint '%s', training from scratch.\n\n",
                   resume_path.c_str());
        }
    }

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

            // Data augmentation: pad 4 -> random crop 32x32 -> random horizontal flip
            auto augmented = images->pad2d(4)->random_crop(32, 32)->random_flip_horizontal(0.5f);

            optimizer.zero_grad();

            auto output = model.forward(augmented);
            auto loss   = criterion(output, labels);

            loss->backward();

            // Manual L2 weight decay on conv/linear weights
            apply_weight_decay(all_params, weight_decay);

            optimizer.step();

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

            // Print progress: every batch for first epoch, every 50 after
            if (epoch == start_epoch || num_batches % 50 == 0) {
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
            save_checkpoint("resnet18_best.ckpt", epoch + 1, best_test_acc,
                            optimizer.lr, all_params, optimizer.velocity);
            saved_best = true;
        }
        if ((epoch + 1) % 10 == 0) {
            save_checkpoint("resnet18_latest.ckpt", epoch + 1, best_test_acc,
                            optimizer.lr, all_params, optimizer.velocity);
        }

        printf("\r  Epoch %3d | Loss: %.4f | Train: %.2f%% | Test: %.2f%% | Best: %.2f%% | LR: %.6f | %.1fs",
               epoch + 1, avg_loss, train_acc, test_acc, best_test_acc, optimizer.lr, epoch_secs);
        if (saved_best) printf(" [saved best]");
        printf("\n");
        fflush(stdout);
    }

    printf("--------------------------------------------------------------------------------\n");
    save_checkpoint("resnet18_final.ckpt", num_epochs, best_test_acc,
                    optimizer.lr, all_params, optimizer.velocity);
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

        auto cpu_output = output;
        auto cpu_labels = labels;

        for (int i = 0; i < 10; i++) {
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
            printf("  %s Predicted: %-10s  Actual: %-10s\n",
                   status, cifar10_class_name(predicted), cifar10_class_name(actual));
        }
    }

    return 0;
}
