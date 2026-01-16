#include <cstdio>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "mnist.h"
#include "dataloader.h"
#include "serialize.h"

// Reshape flat MNIST images to 4D tensor (batch, 1, 28, 28)
TensorPtr reshape_images(const TensorPtr& flat_images) {
    size_t batch = flat_images->shape[0];
    auto reshaped = Tensor::create({batch, 1, 28, 28}, false);
    reshaped->data = flat_images->data;
    return reshaped;
}

float compute_accuracy(Sequential& model, ThreadedDataLoader& loader) {
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0;
    size_t total = 0;

    loader.reset();
    while (loader.has_next()) {
        auto [images, labels] = loader.next_batch_pair();
        auto images_4d = reshape_images(images);
        auto output = model.forward(images_4d);

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

    printf("CNN MNIST Example\n");
    printf("=================\n\n");

    printf("Loading MNIST dataset from '%s'...\n", data_dir.c_str());

    MNISTDataset train_data, test_data;
    try {
        train_data = load_mnist_train(data_dir);
        test_data = load_mnist_test(data_dir);
    } catch (const std::exception& e) {
        printf("Error loading MNIST: %s\n", e.what());
        printf("\nTo download MNIST data, run:\n");
        printf("  mkdir -p %s && cd %s\n", data_dir.c_str(), data_dir.c_str());
        printf("  curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz\n");
        printf("  curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz\n");
        printf("  curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz\n");
        printf("  curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz\n");
        printf("  gunzip *.gz\n");
        return 1;
    }

    printf("Train samples: %zu\n", train_data.num_samples);
    printf("Test samples: %zu\n\n", test_data.num_samples);

    // Hyperparameters
    const size_t batch_size = 64;
    const int num_epochs = 5;
    const float learning_rate = 0.01f;

    // CNN Architecture:
    // Input: (batch, 1, 28, 28)
    // Conv1: (batch, 16, 28, 28) -> Pool: (batch, 16, 14, 14)
    // Conv2: (batch, 32, 14, 14) -> Pool: (batch, 32, 7, 7)
    // Flatten: (batch, 32*7*7) = (batch, 1568)
    // FC: (batch, 10)
    Sequential model({
        // First conv block
        new Conv2d(1, 16, 3, 1, 1),      // 1->16 channels, 3x3 kernel, padding=1
        new BatchNorm2d(16),
        new ReLU(),
        new MaxPool2d(2),                 // 28x28 -> 14x14

        // Second conv block
        new Conv2d(16, 32, 3, 1, 1),     // 16->32 channels
        new BatchNorm2d(32),
        new ReLU(),
        new MaxPool2d(2),                 // 14x14 -> 7x7

        // Classifier
        new Flatten(),                    // 32*7*7 = 1568
        new Linear(32 * 7 * 7, 128),
        new ReLU(),
        new Linear(128, 10)
    });

    CrossEntropyLoss criterion;
    SGD optimizer(model.parameters(), learning_rate, 0.9f);

    // Convert MNISTDataset to generic Dataset for ThreadedDataLoader
    Dataset train_dataset{train_data.images, train_data.labels, train_data.num_samples};
    Dataset test_dataset{test_data.images, test_data.labels, test_data.num_samples};

    // Use ThreadedDataLoader with 2 worker threads for prefetching
    ThreadedDataLoader train_loader(train_dataset, batch_size, true, 2);
    ThreadedDataLoader test_loader(test_dataset, batch_size, false, 0);  // No threading for eval

    // Print model summary with output shapes
    printf("Model Summary:\n");
    model.summary({1, 1, 28, 28});  // MNIST: batch=1, channels=1, 28x28
    printf("\n");

    printf("Optimizer: SGD (lr=%.3f, momentum=0.9)\n", learning_rate);
    printf("Batch size: %zu\n\n", batch_size);

    printf("Training...\n");
    printf("------------------------------------------------------------------\n");

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch_pair();
            auto images_4d = reshape_images(images);

            optimizer.zero_grad();

            auto output = model.forward(images_4d);
            auto loss = criterion(output, labels);

            loss->backward();
            optimizer.step();

            total_loss += loss->data[0];
            num_batches++;

            // Progress indicator
            if (num_batches % 100 == 0) {
                printf("\r  Epoch %d: batch %zu/%zu", epoch + 1, num_batches, train_loader.num_batches());
                fflush(stdout);
            }
        }

        float avg_loss = total_loss / num_batches;
        float train_acc = compute_accuracy(model, train_loader);
        float test_acc = compute_accuracy(model, test_loader);

        printf("\r  Epoch %d | Loss: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%%\n",
               epoch + 1, avg_loss, train_acc, test_acc);
    }

    printf("------------------------------------------------------------------\n");
    printf("Training complete!\n\n");

    // Save the model
    printf("Saving model to 'cnn_mnist.bin'...\n");
    save_model(&model, "cnn_mnist.bin");

    return 0;
}
