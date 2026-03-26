#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "dataloader.h"
#include "serialize.h"

// Load a tensor from the whitematter binary format:
//   magic (4B) | ndim (4B) | shape (ndim * 8B) | float data
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

float compute_accuracy(Sequential& model, ThreadedDataLoader& loader) {
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
    return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) * 100.0f : 0.0f;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data/cats_dogs";
    std::string model_path = "cats_vs_dogs.bin";

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) model_path = argv[2];

    printf("Cats vs Dogs CNN — WhiteMatter\n");
    printf("==============================\n\n");
    fflush(stdout);

    // --- Load dataset ---
    printf("Loading dataset from '%s'...\n", data_dir.c_str());
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
        printf("  python examples/preprocess_cats_dogs.py\n");
        return 1;
    }

    size_t n_train = train_images->shape[0];
    size_t n_test  = test_images->shape[0];
    printf("Train: %zu samples | Test: %zu samples\n", n_train, n_test);
    printf("Image shape: [%zu, %zu, %zu]\n\n",
           train_images->shape[1], train_images->shape[2], train_images->shape[3]);

    // --- Hyperparameters ---
    const size_t batch_size = 32;
    const int num_epochs = 20;
    const float learning_rate = 0.001f;

    // --- Model: 4 conv blocks + GAP + 2 dense ---
    // Input: [batch, 3, 224, 224]
    Sequential model({
        // Block 1: 3 -> 32, 224x224 -> 112x112
        new Conv2d(3, 32, 3, 1, 1),
        new BatchNorm2d(32),
        new ReLU(),
        new MaxPool2d(2),

        // Block 2: 32 -> 64, 112x112 -> 56x56
        new Conv2d(32, 64, 3, 1, 1),
        new BatchNorm2d(64),
        new ReLU(),
        new MaxPool2d(2),

        // Block 3: 64 -> 128, 56x56 -> 28x28
        new Conv2d(64, 128, 3, 1, 1),
        new BatchNorm2d(128),
        new ReLU(),
        new MaxPool2d(2),

        // Block 4: 128 -> 128, 28x28 -> 14x14
        new Conv2d(128, 128, 3, 1, 1),
        new BatchNorm2d(128),
        new ReLU(),
        new MaxPool2d(2),

        // Global average pool: 14x14 -> 1x1
        new AvgPool2d(14),

        // Classifier
        new Flatten(),
        new Linear(128, 128),
        new ReLU(),
        new Dropout(0.3),
        new Linear(128, 2)
    });

    CrossEntropyLoss criterion;
    Adam optimizer(model.parameters(), learning_rate);
    StepLR scheduler(&optimizer, 10, 0.1f);

    // --- DataLoaders ---
    Dataset train_dataset{train_images, train_labels, n_train};
    Dataset test_dataset{test_images, test_labels, n_test};

    ThreadedDataLoader train_loader(train_dataset, batch_size, true, 2);
    ThreadedDataLoader test_loader(test_dataset, batch_size, false, 0);

    // --- Print model info ---
    size_t total_params = 0;
    for (const auto& p : model.parameters()) {
        total_params += p->size();
    }
    printf("Architecture:\n");
    printf("  Conv(3,32,3) -> BN -> ReLU -> MaxPool(2)   [224->112]\n");
    printf("  Conv(32,64,3) -> BN -> ReLU -> MaxPool(2)  [112->56]\n");
    printf("  Conv(64,128,3) -> BN -> ReLU -> MaxPool(2) [56->28]\n");
    printf("  Conv(128,128,3) -> BN -> ReLU -> MaxPool(2)[28->14]\n");
    printf("  AvgPool(14) -> Flatten -> FC(128,128) -> ReLU -> Dropout(0.3) -> FC(128,2)\n");
    printf("  Parameters: %zu (%.1f KB)\n\n", total_params, total_params * 4.0f / 1024);

    printf("Training: Adam lr=%.4f, StepLR(step=10, gamma=0.1), %d epochs, batch %zu\n\n",
           learning_rate, num_epochs, batch_size);
    fflush(stdout);

    // --- Training loop ---
    float best_test_acc = 0.0f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch_pair();

            // Data augmentation: random horizontal flip
            auto augmented = images->random_flip_horizontal(0.5f);

            optimizer.zero_grad();
            auto output = model.forward(augmented);
            auto loss = criterion(output, labels);
            loss->backward();
            auto params = model.parameters();
            clip_grad_norm_(params, 5.0f);
            optimizer.step();

            total_loss += loss->data()[0];
            num_batches++;

            if (num_batches % 20 == 0) {
                printf("\r  Epoch %2d: batch %3zu/%zu",
                       epoch + 1, num_batches, train_loader.num_batches());
                fflush(stdout);
            }
        }

        scheduler.step();

        float avg_loss = total_loss / num_batches;
        float test_acc = compute_accuracy(model, test_loader);

        if (test_acc > best_test_acc) {
            best_test_acc = test_acc;
            save_model(&model, model_path);
        }

        printf("\r  Epoch %2d | Loss: %.4f | Test Acc: %.2f%% | Best: %.2f%% | LR: %.6f\n",
               epoch + 1, avg_loss, test_acc, best_test_acc, optimizer.lr);
        fflush(stdout);
    }

    printf("\nDone! Best accuracy: %.2f%% | Model saved to '%s'\n", best_test_acc, model_path.c_str());

    // --- Sample predictions ---
    const char* class_names[] = {"cat", "dog"};
    printf("\nSample predictions:\n");
    {
        NoGradGuard no_grad;
        model.eval();
        test_loader.reset();
        auto [images, labels] = test_loader.next_batch_pair();
        auto output = model.forward(images);

        for (int i = 0; i < 10 && i < static_cast<int>(output->shape[0]); i++) {
            size_t predicted = output->data()[i * 2 + 1] > output->data()[i * 2] ? 1 : 0;
            int actual = static_cast<int>(labels->data()[i]);
            const char* status = (predicted == static_cast<size_t>(actual)) ? "OK" : "XX";
            printf("  [%s] Predicted: %-4s  Actual: %-4s\n",
                   status, class_names[predicted], class_names[actual]);
        }
    }

    return 0;
}
