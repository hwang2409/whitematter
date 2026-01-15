#include <cstdio>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "cifar10.h"
#include "serialize.h"

float compute_accuracy(Sequential& model, CIFAR10DataLoader& loader) {
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0;
    size_t total = 0;

    loader.reset();
    while (loader.has_next()) {
        auto [images, labels] = loader.next_batch();
        auto output = model.forward(images);

        size_t batch_size = output->shape[0];
        size_t num_classes = output->shape[1];

        for (size_t i = 0; i < batch_size; i++) {
            size_t predicted = 0;
            float max_val = output->data[i * num_classes];
            for (size_t j = 1; j < num_classes; j++) {
                if (output->data[i * num_classes + j] > max_val) {
                    max_val = output->data[i * num_classes + j];
                    predicted = j;
                }
            }
            if (predicted == static_cast<size_t>(labels->data[i])) {
                correct++;
            }
            total++;
        }
    }

    model.train();
    return static_cast<float>(correct) / static_cast<float>(total) * 100.0f;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    printf("CNN CIFAR-10 Example\n");
    printf("====================\n\n");
    fflush(stdout);

    printf("Loading CIFAR-10 dataset from '%s'...\n", data_dir.c_str());
    fflush(stdout);

    CIFAR10Dataset train_data, test_data;
    try {
        train_data = load_cifar10_train(data_dir);
        test_data = load_cifar10_test(data_dir);
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

    // Hyperparameters
    const size_t batch_size = 128;
    const int num_epochs = 50;
    const float learning_rate = 0.001f;

    // VGG-style CNN Architecture for CIFAR-10:
    // Input: (batch, 3, 32, 32)
    // Block 1: Conv64x2 -> MaxPool -> (batch, 64, 16, 16)
    // Block 2: Conv128x2 -> MaxPool -> (batch, 128, 8, 8)
    // Block 3: Conv256x2 -> MaxPool -> (batch, 256, 4, 4)
    // Flatten: (batch, 256*4*4) = (batch, 4096)
    // FC: 4096 -> 512 -> 10

    Sequential model({
        // Block 1: 3 -> 64 channels
        new Conv2d(3, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new Conv2d(64, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new MaxPool2d(2),                 // 32x32 -> 16x16

        // Block 2: 64 -> 128 channels
        new Conv2d(64, 128, 3, 1, 1),
        new BatchNorm2d(128),
        new ReLU(),
        new Conv2d(128, 128, 3, 1, 1),
        new BatchNorm2d(128),
        new ReLU(),
        new MaxPool2d(2),                 // 16x16 -> 8x8

        // Block 3: 128 -> 256 channels
        new Conv2d(128, 256, 3, 1, 1),
        new BatchNorm2d(256),
        new ReLU(),
        new Conv2d(256, 256, 3, 1, 1),
        new BatchNorm2d(256),
        new ReLU(),
        new MaxPool2d(2),                 // 8x8 -> 4x4

        // Classifier
        new Flatten(),                    // 256*4*4 = 4096
        new Linear(256 * 4 * 4, 512),
        new ReLU(),
        new Dropout(0.5),
        new Linear(512, 10)
    });

    CrossEntropyLoss criterion;
    Adam optimizer(model.parameters(), learning_rate);
    CosineAnnealingLR scheduler(&optimizer, num_epochs, 1e-6f);

    // Data loaders with augmentation for training
    CIFAR10DataLoader train_loader(train_data, batch_size, true, true);   // shuffle=true, augment=true
    CIFAR10DataLoader test_loader(test_data, batch_size, false, false);   // no shuffle, no augment

    // Count parameters
    size_t total_params = 0;
    for (const auto& p : model.parameters()) {
        total_params += p->size();
    }

    printf("Model Architecture (VGG-style CNN):\n");
    printf("  Block 1: Conv(3,64,3) -> BN -> ReLU -> Conv(64,64,3) -> BN -> ReLU -> MaxPool(2)\n");
    printf("  Block 2: Conv(64,128,3) -> BN -> ReLU -> Conv(128,128,3) -> BN -> ReLU -> MaxPool(2)\n");
    printf("  Block 3: Conv(128,256,3) -> BN -> ReLU -> Conv(256,256,3) -> BN -> ReLU -> MaxPool(2)\n");
    printf("  Flatten -> Linear(4096,512) -> ReLU -> Dropout(0.5) -> Linear(512,10)\n");
    printf("  Total parameters: %zu (~%.2f KB)\n\n", total_params, total_params * 4.0f / 1024);

    printf("Optimizer: Adam (lr=%.4f)\n", learning_rate);
    printf("Scheduler: CosineAnnealingLR (T_max=%d)\n", num_epochs);
    printf("Batch size: %zu\n", batch_size);
    printf("Data augmentation: random crop (pad=4) + horizontal flip\n\n");
    fflush(stdout);

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
            auto [images, labels] = train_loader.next_batch();

            optimizer.zero_grad();

            auto output = model.forward(images);
            auto loss = criterion(output, labels);

            loss->backward();
            optimizer.step();

            total_loss += loss->data[0];
            num_batches++;

            // Progress indicator
            if (num_batches % 50 == 0) {
                printf("\r  Epoch %2d: batch %3zu/%zu (lr=%.6f)",
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

        printf("\r  Epoch %2d | Loss: %.4f | Test Acc: %.2f%% | Best: %.2f%%\n",
               epoch + 1, avg_loss, test_acc, best_test_acc);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Training complete! Best test accuracy: %.2f%%\n\n", best_test_acc);

    // Save the model
    printf("Saving model to 'cnn_cifar10.bin'...\n");
    save_model(&model, "cnn_cifar10.bin");

    // Print some predictions
    printf("\nSample predictions:\n");
    {
        NoGradGuard no_grad;
        model.eval();
        test_loader.reset();
        auto [images, labels] = test_loader.next_batch();
        auto output = model.forward(images);

        for (int i = 0; i < 10; i++) {
            size_t predicted = 0;
            float max_val = output->data[i * 10];
            for (size_t j = 1; j < 10; j++) {
                if (output->data[i * 10 + j] > max_val) {
                    max_val = output->data[i * 10 + j];
                    predicted = j;
                }
            }
            int actual = static_cast<int>(labels->data[i]);
            const char* status = (predicted == static_cast<size_t>(actual)) ? "✓" : "✗";
            printf("  %s Predicted: %-10s  Actual: %-10s\n",
                   status, cifar10_class_name(predicted), cifar10_class_name(actual));
        }
    }

    return 0;
}
