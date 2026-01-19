#include <cstdio>
#include <cstring>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "mnist.h"
#include "dataloader.h"
#include "model_zoo.h"

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

void print_usage() {
    printf("Usage: train_zoo_model <model_name> [data_dir] [epochs]\n\n");
    printf("Available models:\n");
    for (const auto& name : list_models()) {
        auto info = ModelZoo::instance().get_info(name);
        printf("  %-16s - %s\n", name.c_str(), info.description.c_str());
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string model_name = argv[1];
    std::string data_dir = argc > 2 ? argv[2] : "data";
    int num_epochs = argc > 3 ? atoi(argv[3]) : 5;

    // Check if model exists
    if (!ModelZoo::instance().has_model(model_name)) {
        printf("Error: Unknown model '%s'\n\n", model_name.c_str());
        print_usage();
        return 1;
    }

    auto info = ModelZoo::instance().get_info(model_name);
    printf("Training Model Zoo: %s\n", model_name.c_str());
    printf("==================================\n\n");
    printf("Description: %s\n", info.description.c_str());
    printf("Dataset: %s\n", info.dataset.c_str());
    printf("Parameters: %zu\n", info.num_params);
    printf("Expected accuracy: %.1f%%\n\n", info.accuracy);

    // Currently only MNIST models supported
    if (info.dataset != "MNIST") {
        printf("Error: Only MNIST models supported in this trainer.\n");
        printf("       Use cnn_cifar10 for CIFAR-10 models.\n");
        return 1;
    }

    // Load data
    printf("Loading MNIST from '%s'...\n", data_dir.c_str());
    MNISTDataset train_data, test_data;
    try {
        train_data = load_mnist_train(data_dir);
        test_data = load_mnist_test(data_dir);
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }
    printf("Train: %zu, Test: %zu\n\n", train_data.num_samples, test_data.num_samples);

    // Create model from zoo
    printf("Creating model from zoo...\n");
    Sequential* model = create_model(model_name);
    if (!model) {
        printf("Error: Failed to create model\n");
        return 1;
    }

    // Setup training
    const size_t batch_size = 64;
    const float learning_rate = 0.01f;

    CrossEntropyLoss criterion;
    SGD optimizer(model->parameters(), learning_rate, 0.9f);

    Dataset train_dataset{train_data.images, train_data.labels, train_data.num_samples};
    Dataset test_dataset{test_data.images, test_data.labels, test_data.num_samples};

    ThreadedDataLoader train_loader(train_dataset, batch_size, true, 2);
    ThreadedDataLoader test_loader(test_dataset, batch_size, false, 0);

    printf("Training for %d epochs (lr=%.3f, batch=%zu)...\n", num_epochs, learning_rate, batch_size);
    printf("--------------------------------------------------\n");

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch_pair();
            auto images_4d = reshape_images(images);

            optimizer.zero_grad();
            auto output = model->forward(images_4d);
            auto loss = criterion(output, labels);
            loss->backward();
            optimizer.step();

            total_loss += loss->data[0];
            num_batches++;

            if (num_batches % 100 == 0) {
                printf("\r  Epoch %d: batch %zu/%zu", epoch + 1, num_batches, train_loader.num_batches());
                fflush(stdout);
            }
        }

        float avg_loss = total_loss / num_batches;
        float test_acc = compute_accuracy(*model, test_loader);

        printf("\r  Epoch %d | Loss: %.4f | Test Acc: %.2f%%\n", epoch + 1, avg_loss, test_acc);
    }

    printf("--------------------------------------------------\n\n");

    // Save to zoo
    printf("Saving to model zoo...\n");
    ModelZoo::instance().save_to_zoo(model_name, model);

    // Verify
    float final_acc = compute_accuracy(*model, test_loader);
    printf("\nFinal test accuracy: %.2f%%\n", final_acc);
    printf("Model saved to: pretrained/%s.bin\n", model_name.c_str());

    delete model;
    return 0;
}
