#include <iostream>
#include <cstdio>

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "mnist.h"
#include "serialize.h"

float compute_accuracy(Sequential& model, DataLoader& loader) {
    NoGradGuard no_grad;  // Disable gradient tracking for inference

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

    return static_cast<float>(correct) / static_cast<float>(total) * 100.0f;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    printf("Neural Network Framework Demo\n");
    printf("==============================\n\n");

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

    const size_t batch_size = 64;
    const int num_epochs = 3;
    const float learning_rate = 0.01f;

    Sequential model({
        new Linear(784, 256),
        new ReLU(),
        new Linear(256, 128),
        new ReLU(),
        new Linear(128, 10)
    });

    CrossEntropyLoss criterion;
    SGD optimizer(model.parameters(), learning_rate, 0.9f);

    DataLoader train_loader(train_data, batch_size, true);
    DataLoader test_loader(test_data, batch_size, false);

    printf("Model: Linear(784->256) -> ReLU -> Linear(256->128) -> ReLU -> Linear(128->10)\n");
    printf("Optimizer: SGD (lr=%.3f, momentum=0.9)\n", learning_rate);
    printf("Batch size: %zu\n\n", batch_size);

    printf("Training...\n");
    printf("---------------------------------------------------------------\n");

    for (int epoch = 0; epoch < num_epochs; epoch++) {
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
        }

        float avg_loss = total_loss / num_batches;
        float train_acc = compute_accuracy(model, train_loader);
        float test_acc = compute_accuracy(model, test_loader);

        printf("Epoch %2d | Loss: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%%\n",
               epoch + 1, avg_loss, train_acc, test_acc);
    }

    printf("---------------------------------------------------------------\n");
    printf("Training complete!\n\n");

    // Save the trained model
    printf("Saving model...\n");
    save_model(&model, "model.bin");

    // Save checkpoint (model + optimizer state)
    float final_test_acc = compute_accuracy(model, test_loader) / 100.0f;
    save_checkpoint("checkpoint.bin", &model, &optimizer, num_epochs, 0.0f, final_test_acc);

    // Demonstrate loading: create a new model and load weights
    printf("\nDemonstrating model loading...\n");
    Sequential loaded_model({
        new Linear(784, 256),
        new ReLU(),
        new Linear(256, 128),
        new ReLU(),
        new Linear(128, 10)
    });

    load_model(&loaded_model, "model.bin");

    // Verify loaded model works
    float loaded_acc = compute_accuracy(loaded_model, test_loader);
    printf("Loaded model test accuracy: %.2f%%\n", loaded_acc);

    return 0;
}
