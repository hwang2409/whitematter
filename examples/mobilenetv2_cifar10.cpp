#include <cstdio>
#include <vector>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "cifar10.h"
#include "dataloader.h"
#include "serialize.h"

// ---------------------------------------------------------------------------
// InvertedResidual block for MobileNetV2
// ---------------------------------------------------------------------------
struct InvertedResidual {
    bool use_residual;
    int expand_ratio;

    // Expansion phase (only when expand_ratio != 1)
    Conv2d*       expand_conv = nullptr;
    BatchNorm2d*  expand_bn   = nullptr;
    SiLU*         expand_act  = nullptr;

    // Depthwise separable convolution
    Conv2d*       dw_conv;
    BatchNorm2d*  dw_bn;
    SiLU*         dw_act;

    // Projection (linear bottleneck — no activation)
    Conv2d*       proj_conv;
    BatchNorm2d*  proj_bn;

    InvertedResidual(size_t in_ch, size_t out_ch, size_t stride, size_t t) {
        expand_ratio = static_cast<int>(t);
        use_residual = (stride == 1 && in_ch == out_ch);
        size_t expanded_ch = in_ch * t;

        // Expansion 1x1
        if (t != 1) {
            expand_conv = new Conv2d(in_ch, expanded_ch, 1, 1, 0);
            expand_bn   = new BatchNorm2d(expanded_ch);
            expand_act  = new SiLU();
        }

        // Depthwise 3x3 (groups = expanded_ch)
        dw_conv = new Conv2d(expanded_ch, expanded_ch, 3, stride, 1, expanded_ch);
        dw_bn   = new BatchNorm2d(expanded_ch);
        dw_act  = new SiLU();

        // Projection 1x1 (no activation)
        proj_conv = new Conv2d(expanded_ch, out_ch, 1, 1, 0);
        proj_bn   = new BatchNorm2d(out_ch);
    }

    ~InvertedResidual() {
        delete expand_conv;
        delete expand_bn;
        delete expand_act;
        delete dw_conv;
        delete dw_bn;
        delete dw_act;
        delete proj_conv;
        delete proj_bn;
    }

    TensorPtr forward(const TensorPtr& input) {
        TensorPtr x = input;

        // Expansion
        if (expand_ratio != 1) {
            x = expand_conv->forward(x);
            x = expand_bn->forward(x);
            x = expand_act->forward(x);
        }

        // Depthwise
        x = dw_conv->forward(x);
        x = dw_bn->forward(x);
        x = dw_act->forward(x);

        // Projection (linear — no activation)
        x = proj_conv->forward(x);
        x = proj_bn->forward(x);

        // Residual connection
        if (use_residual) {
            x = x->add(input);
        }

        return x;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto append = [&](std::vector<TensorPtr> p) {
            params.insert(params.end(), p.begin(), p.end());
        };
        if (expand_conv) {
            append(expand_conv->parameters());
            append(expand_bn->parameters());
        }
        append(dw_conv->parameters());
        append(dw_bn->parameters());
        append(proj_conv->parameters());
        append(proj_bn->parameters());
        return params;
    }

    void train() {
        if (expand_bn) expand_bn->train();
        dw_bn->train();
        proj_bn->train();
    }

    void eval() {
        if (expand_bn) expand_bn->eval();
        dw_bn->eval();
        proj_bn->eval();
    }
};

// ---------------------------------------------------------------------------
// MobileNetV2 for CIFAR-10 (lighter version for 32x32 images)
// ---------------------------------------------------------------------------
struct MobileNetV2 {
    // Initial convolution: 3 -> 32
    Conv2d*       init_conv;
    BatchNorm2d*  init_bn;
    SiLU*         init_act;

    // Inverted residual blocks
    std::vector<InvertedResidual*> blocks;

    // Final convolution: 96 -> 256
    Conv2d*       final_conv;
    BatchNorm2d*  final_bn;
    SiLU*         final_act;

    // Classifier
    AvgPool2d*    pool;
    Flatten*      flatten;
    Linear*       classifier;

    MobileNetV2() {
        // Initial conv: 3 -> 32, 32x32
        init_conv = new Conv2d(3, 32, 3, 1, 1);
        init_bn   = new BatchNorm2d(32);
        init_act  = new SiLU();

        // Inverted residuals (CIFAR-10 lightweight config)
        //                          in   out  stride  expand_ratio
        blocks.push_back(new InvertedResidual(32,  16,  1,  1));  // 32x32

        blocks.push_back(new InvertedResidual(16,  24,  2,  6));  // 16x16
        blocks.push_back(new InvertedResidual(24,  24,  1,  6));  // 16x16

        blocks.push_back(new InvertedResidual(24,  32,  2,  6));  // 8x8
        blocks.push_back(new InvertedResidual(32,  32,  1,  6));  // 8x8

        blocks.push_back(new InvertedResidual(32,  64,  2,  6));  // 4x4
        blocks.push_back(new InvertedResidual(64,  64,  1,  6));  // 4x4

        blocks.push_back(new InvertedResidual(64,  96,  1,  6));  // 4x4

        // Final conv: 96 -> 256
        final_conv = new Conv2d(96, 256, 1, 1, 0);
        final_bn   = new BatchNorm2d(256);
        final_act  = new SiLU();

        // Classifier
        pool      = new AvgPool2d(4);
        flatten   = new Flatten();
        classifier = new Linear(256, 10);
    }

    ~MobileNetV2() {
        delete init_conv;
        delete init_bn;
        delete init_act;
        for (auto* b : blocks) delete b;
        delete final_conv;
        delete final_bn;
        delete final_act;
        delete pool;
        delete flatten;
        delete classifier;
    }

    TensorPtr forward(const TensorPtr& input) {
        // Initial conv
        auto x = init_conv->forward(input);
        x = init_bn->forward(x);
        x = init_act->forward(x);

        // Inverted residuals
        for (auto* block : blocks) {
            x = block->forward(x);
        }

        // Final conv
        x = final_conv->forward(x);
        x = final_bn->forward(x);
        x = final_act->forward(x);

        // Pool + classify
        x = pool->forward(x);
        x = flatten->forward(x);
        x = classifier->forward(x);

        return x;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto append = [&](std::vector<TensorPtr> p) {
            params.insert(params.end(), p.begin(), p.end());
        };

        append(init_conv->parameters());
        append(init_bn->parameters());

        for (auto* block : blocks) {
            append(block->parameters());
        }

        append(final_conv->parameters());
        append(final_bn->parameters());

        append(classifier->parameters());
        return params;
    }

    void train() {
        init_bn->train();
        for (auto* block : blocks) block->train();
        final_bn->train();
    }

    void eval() {
        init_bn->eval();
        for (auto* block : blocks) block->eval();
        final_bn->eval();
    }

    size_t num_parameters() const {
        size_t total = 0;
        for (auto& p : const_cast<MobileNetV2*>(this)->parameters()) {
            total += p->size();
        }
        return total;
    }
};

// ---------------------------------------------------------------------------
// Accuracy evaluation
// ---------------------------------------------------------------------------
float compute_accuracy(MobileNetV2& model, ThreadedDataLoader& loader) {
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0;
    size_t total = 0;

    loader.reset();
    while (loader.has_next()) {
        auto [images, labels] = loader.next_batch_pair();
        auto output = model.forward(images);

        size_t batch_size = output->shape[0];
        size_t num_classes = output->shape[1];

        for (size_t i = 0; i < batch_size; i++) {
            size_t predicted = 0;
            float max_val = output->data()[i * num_classes];
            for (size_t j = 1; j < num_classes; j++) {
                if (output->data()[i * num_classes + j] > max_val) {
                    max_val = output->data()[i * num_classes + j];
                    predicted = j;
                }
            }
            if (predicted == static_cast<size_t>(labels->data()[i])) {
                correct++;
            }
            total++;
        }
    }

    model.train();
    return static_cast<float>(correct) / static_cast<float>(total) * 100.0f;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    printf("MobileNetV2 CIFAR-10 Example\n");
    printf("=============================\n\n");
    fflush(stdout);

    // -----------------------------------------------------------------------
    // Load dataset
    // -----------------------------------------------------------------------
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
    printf("Test samples: %zu\n\n", test_data.num_samples);

    // -----------------------------------------------------------------------
    // Hyperparameters
    // -----------------------------------------------------------------------
    const size_t batch_size    = 128;
    const int    num_epochs    = 200;
    const float  learning_rate = 3e-3f;

    // -----------------------------------------------------------------------
    // Build model
    // -----------------------------------------------------------------------
    MobileNetV2 model;

    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& p : params) {
        total_params += p->size();
    }

    printf("Model: MobileNetV2 (CIFAR-10 lightweight)\n");
    printf("  Conv2d(3,32,3) -> BN -> SiLU                  [32x32]\n");
    printf("  InvRes(32,16,  s=1, t=1)                      [32x32]\n");
    printf("  InvRes(16,24,  s=2, t=6) x2                   [16x16]\n");
    printf("  InvRes(24,32,  s=2, t=6) x2                   [8x8]\n");
    printf("  InvRes(32,64,  s=2, t=6) x2                   [4x4]\n");
    printf("  InvRes(64,96,  s=1, t=6)                      [4x4]\n");
    printf("  Conv2d(96,256,1) -> BN -> SiLU                [4x4]\n");
    printf("  AvgPool(4) -> Flatten -> Linear(256,10)\n");
    printf("  Total parameters: %zu (~%.2f KB)\n\n", total_params, total_params * 4.0f / 1024);

    // -----------------------------------------------------------------------
    // Optimizer & scheduler
    // -----------------------------------------------------------------------
    Adam optimizer(params, learning_rate);
    CosineAnnealingLR scheduler(&optimizer, num_epochs, 1e-6f);
    CrossEntropyLoss criterion;

    printf("Optimizer: Adam (lr=%.4f)\n", learning_rate);
    printf("Scheduler: CosineAnnealingLR (T_max=%d, eta_min=1e-6)\n", num_epochs);
    printf("Batch size: %zu\n", batch_size);
    printf("Epochs: %d\n", num_epochs);
    printf("Data augmentation: pad(4) + random_crop(32,32) + random_flip_horizontal(0.5)\n\n");
    fflush(stdout);

    // -----------------------------------------------------------------------
    // Data loaders
    // -----------------------------------------------------------------------
    Dataset train_dataset{train_data.images, train_data.labels, train_data.num_samples};
    Dataset test_dataset{test_data.images, test_data.labels, test_data.num_samples};

    ThreadedDataLoader train_loader(train_dataset, batch_size, true, 2);
    ThreadedDataLoader test_loader(test_dataset, batch_size, false, 0);

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------
    printf("Training...\n");
    printf("--------------------------------------------------------------------------------\n");
    fflush(stdout);

    float best_test_acc = 0.0f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch_pair();

            // Data augmentation: pad(4) -> random_crop(32,32) -> random_flip_horizontal
            auto augmented = images->pad2d(4)->random_crop(32, 32)->random_flip_horizontal(0.5f);

            optimizer.zero_grad();

            auto output = model.forward(augmented);
            auto loss = criterion(output, labels);

            loss->backward();
            optimizer.step();

            total_loss += loss->data()[0];
            num_batches++;

            if (num_batches % 50 == 0) {
                printf("\r  Epoch %3d: batch %3zu/%zu (lr=%.6f)",
                       epoch + 1, num_batches, train_loader.num_batches(), optimizer.lr);
                fflush(stdout);
            }
        }

        scheduler.step();

        float avg_loss = total_loss / num_batches;
        float test_acc = compute_accuracy(model, test_loader);

        if (test_acc > best_test_acc) {
            best_test_acc = test_acc;
        }

        printf("\r  Epoch %3d | Loss: %.4f | Test Acc: %.2f%% | Best: %.2f%% | LR: %.6f\n",
               epoch + 1, avg_loss, test_acc, best_test_acc, optimizer.lr);
        fflush(stdout);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Training complete! Best test accuracy: %.2f%%\n\n", best_test_acc);

    // -----------------------------------------------------------------------
    // Save model
    // -----------------------------------------------------------------------
    printf("Saving model to 'mobilenetv2_cifar10.bin'...\n");
    fflush(stdout);

    // Save as a flat parameter dump
    {
        auto all_params = model.parameters();
        std::ofstream out("mobilenetv2_cifar10.bin", std::ios::binary);
        uint32_t magic = 0x574D4D00;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        uint32_t n = static_cast<uint32_t>(all_params.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (auto& p : all_params) {
            save_tensor(p, out);
        }
        out.close();
        printf("Model saved (%u parameter tensors).\n", n);
    }

    // -----------------------------------------------------------------------
    // Sample predictions
    // -----------------------------------------------------------------------
    printf("\nSample predictions:\n");
    {
        NoGradGuard no_grad;
        model.eval();
        test_loader.reset();
        auto [images, labels] = test_loader.next_batch_pair();
        auto output = model.forward(images);

        for (int i = 0; i < 10; i++) {
            size_t predicted = 0;
            float max_val = output->data()[i * 10];
            for (size_t j = 1; j < 10; j++) {
                if (output->data()[i * 10 + j] > max_val) {
                    max_val = output->data()[i * 10 + j];
                    predicted = j;
                }
            }
            int actual = static_cast<int>(labels->data()[i]);
            const char* status = (predicted == static_cast<size_t>(actual)) ? "OK" : "MISS";
            printf("  [%4s] Predicted: %-10s  Actual: %-10s\n",
                   status, cifar10_class_name(predicted), cifar10_class_name(actual));
        }
    }

    return 0;
}
