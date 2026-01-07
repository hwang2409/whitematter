#ifndef MNIST_H
#define MNIST_H

#include "tensor.h"
#include <string>
#include <utility>

struct MNISTDataset {
    TensorPtr images;
    TensorPtr labels;
    size_t num_samples;
};

MNISTDataset load_mnist_train(const std::string& data_dir);
MNISTDataset load_mnist_test(const std::string& data_dir);

TensorPtr load_mnist_images(const std::string& filepath);
TensorPtr load_mnist_labels(const std::string& filepath);

class DataLoader {
public:
    TensorPtr images;
    TensorPtr labels;
    size_t batch_size;
    size_t num_samples;
    size_t current_idx;
    bool shuffle;
    std::vector<size_t> indices;

    DataLoader(const MNISTDataset& dataset, size_t batch_size, bool shuffle = true);

    void reset();
    bool has_next() const;
    std::pair<TensorPtr, TensorPtr> next_batch();
    size_t num_batches() const;
};

#endif
