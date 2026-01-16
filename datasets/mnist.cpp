#include "mnist.h"
#include <fstream>
#include <algorithm>
#include <random>
#include <stdexcept>

static uint32_t read_uint32_be(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

TensorPtr load_mnist_images(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t num_images = read_uint32_be(file);
    uint32_t num_rows = read_uint32_be(file);
    uint32_t num_cols = read_uint32_be(file);

    size_t image_size = num_rows * num_cols;
    auto images = Tensor::create({num_images, image_size}, false);

    std::vector<unsigned char> buffer(image_size);
    for (uint32_t i = 0; i < num_images; i++) {
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        for (size_t j = 0; j < image_size; j++) {
            images->data[i * image_size + j] = buffer[j] / 255.0f;
        }
    }

    return images;
}

TensorPtr load_mnist_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t num_labels = read_uint32_be(file);
    auto labels = Tensor::create({num_labels}, false);

    std::vector<unsigned char> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);
    for (uint32_t i = 0; i < num_labels; i++) {
        labels->data[i] = static_cast<float>(buffer[i]);
    }

    return labels;
}

MNISTDataset load_mnist_train(const std::string& data_dir) {
    MNISTDataset dataset;
    dataset.images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    dataset.labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    dataset.num_samples = dataset.images->shape[0];
    return dataset;
}

MNISTDataset load_mnist_test(const std::string& data_dir) {
    MNISTDataset dataset;
    dataset.images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    dataset.labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");
    dataset.num_samples = dataset.images->shape[0];
    return dataset;
}

DataLoader::DataLoader(const MNISTDataset& dataset, size_t batch_size, bool shuffle)
    : images(dataset.images), labels(dataset.labels), batch_size(batch_size),
      num_samples(dataset.num_samples), current_idx(0), shuffle(shuffle) {
    indices.resize(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        indices[i] = i;
    }
    if (shuffle) {
        reset();
    }
}

void DataLoader::reset() {
    current_idx = 0;
    if (shuffle) {
        static std::mt19937 rng(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
}

bool DataLoader::has_next() const {
    return current_idx < num_samples;
}

std::pair<TensorPtr, TensorPtr> DataLoader::next_batch() {
    size_t end_idx = std::min(current_idx + batch_size, num_samples);
    size_t actual_batch_size = end_idx - current_idx;
    size_t image_size = images->shape[1];

    auto batch_images = Tensor::create({actual_batch_size, image_size}, false);
    auto batch_labels = Tensor::create({actual_batch_size}, false);

    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = indices[current_idx + i];
        for (size_t j = 0; j < image_size; j++) {
            batch_images->data[i * image_size + j] = images->data[idx * image_size + j];
        }
        batch_labels->data[i] = labels->data[idx];
    }

    current_idx = end_idx;
    return {batch_images, batch_labels};
}

size_t DataLoader::num_batches() const {
    return (num_samples + batch_size - 1) / batch_size;
}
