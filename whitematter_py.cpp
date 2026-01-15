#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "layer.h"
#include "serialize.h"

namespace py = pybind11;

// Build CIFAR-10 VGG-style model into a Sequential
void build_cifar10_model(Sequential& model) {
    // Block 1: 3 -> 64 channels
    model.add(new Conv2d(3, 64, 3, 1, 1));
    model.add(new BatchNorm2d(64));
    model.add(new ReLU());
    model.add(new Conv2d(64, 64, 3, 1, 1));
    model.add(new BatchNorm2d(64));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    // Block 2: 64 -> 128 channels
    model.add(new Conv2d(64, 128, 3, 1, 1));
    model.add(new BatchNorm2d(128));
    model.add(new ReLU());
    model.add(new Conv2d(128, 128, 3, 1, 1));
    model.add(new BatchNorm2d(128));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    // Block 3: 128 -> 256 channels
    model.add(new Conv2d(128, 256, 3, 1, 1));
    model.add(new BatchNorm2d(256));
    model.add(new ReLU());
    model.add(new Conv2d(256, 256, 3, 1, 1));
    model.add(new BatchNorm2d(256));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    // Classifier
    model.add(new Flatten());
    model.add(new Linear(256 * 4 * 4, 512));
    model.add(new ReLU());
    model.add(new Dropout(0.5));
    model.add(new Linear(512, 10));
}

// Build simple 2-layer CNN into a Sequential
void build_cifar10_simple_model(Sequential& model) {
    model.add(new Conv2d(3, 32, 3, 1, 1));
    model.add(new BatchNorm2d(32));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    model.add(new Conv2d(32, 64, 3, 1, 1));
    model.add(new BatchNorm2d(64));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    model.add(new Flatten());
    model.add(new Linear(64 * 8 * 8, 256));
    model.add(new ReLU());
    model.add(new Dropout(0.5));
    model.add(new Linear(256, 10));
}

// Build MNIST CNN model
void build_mnist_model(Sequential& model) {
    model.add(new Conv2d(1, 16, 3, 1, 1));
    model.add(new BatchNorm2d(16));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    model.add(new Conv2d(16, 32, 3, 1, 1));
    model.add(new BatchNorm2d(32));
    model.add(new ReLU());
    model.add(new MaxPool2d(2));

    model.add(new Flatten());
    model.add(new Linear(32 * 7 * 7, 128));
    model.add(new ReLU());
    model.add(new Linear(128, 10));
}

class ModelWrapper {
public:
    std::unique_ptr<Sequential> model;
    bool loaded = false;
    std::string model_type;

    ModelWrapper() {}

    void load(const std::string& path, const std::string& arch = "auto") {
        // Create appropriate model architecture
        model = std::make_unique<Sequential>();

        if (arch == "vgg" || arch == "cifar10") {
            build_cifar10_model(*model);
            model_type = "vgg";
        } else if (arch == "simple" || arch == "simple_cnn") {
            build_cifar10_simple_model(*model);
            model_type = "simple";
        } else if (arch == "mnist" || arch == "mnist_cnn") {
            build_mnist_model(*model);
            model_type = "mnist";
        } else {
            // Auto-detect: try VGG first
            build_cifar10_model(*model);
            model_type = "vgg";
        }

        // Load weights
        if (!load_model(model.get(), path)) {
            // Try other models if auto fails
            if (arch == "auto") {
                // Try simple CIFAR model
                model = std::make_unique<Sequential>();
                build_cifar10_simple_model(*model);
                model_type = "simple";
                if (!load_model(model.get(), path)) {
                    // Try MNIST model
                    model = std::make_unique<Sequential>();
                    build_mnist_model(*model);
                    model_type = "mnist";
                    if (!load_model(model.get(), path)) {
                        throw std::runtime_error("Failed to load model from: " + path);
                    }
                }
            } else {
                throw std::runtime_error("Failed to load model from: " + path);
            }
        }

        model->eval();
        loaded = true;
    }

    // Predict from numpy array [C, H, W] or [N, C, H, W]
    py::array_t<float> predict(py::array_t<float> input) {
        if (!loaded) {
            throw std::runtime_error("Model not loaded. Call load() first.");
        }

        NoGradGuard no_grad;
        auto buf = input.request();
        float* ptr = static_cast<float*>(buf.ptr);

        // Determine shape
        std::vector<size_t> shape;
        for (auto dim : buf.shape) {
            shape.push_back(static_cast<size_t>(dim));
        }

        // Add batch dimension if needed [C,H,W] -> [1,C,H,W]
        bool added_batch = false;
        if (shape.size() == 3) {
            shape.insert(shape.begin(), 1);
            added_batch = true;
        }

        // Create tensor and copy data
        auto tensor = Tensor::create(shape, false);
        std::copy(ptr, ptr + tensor->size(), tensor->data.begin());

        // Forward pass
        auto output = model->forward(tensor);

        // Return as numpy array
        std::vector<ssize_t> out_shape;
        for (auto dim : output->shape) {
            out_shape.push_back(static_cast<ssize_t>(dim));
        }

        auto result = py::array_t<float>(out_shape);
        auto result_buf = result.request();
        float* result_ptr = static_cast<float*>(result_buf.ptr);
        std::copy(output->data.begin(), output->data.end(), result_ptr);

        return result;
    }

    // Get predicted class index
    int predict_class(py::array_t<float> input) {
        auto output = predict(input);
        auto buf = output.request();
        float* ptr = static_cast<float*>(buf.ptr);

        // Find argmax of last dimension
        size_t num_classes = buf.shape[buf.ndim - 1];
        size_t offset = (buf.ndim > 1) ? 0 : 0;  // First sample if batched

        int max_idx = 0;
        float max_val = ptr[offset];
        for (size_t i = 1; i < num_classes; i++) {
            if (ptr[offset + i] > max_val) {
                max_val = ptr[offset + i];
                max_idx = static_cast<int>(i);
            }
        }
        return max_idx;
    }

    // Get probabilities (softmax)
    py::array_t<float> predict_proba(py::array_t<float> input) {
        auto output = predict(input);
        auto buf = output.request();
        float* ptr = static_cast<float*>(buf.ptr);

        size_t batch_size = (buf.ndim > 1) ? buf.shape[0] : 1;
        size_t num_classes = buf.shape[buf.ndim - 1];

        // Apply softmax
        for (size_t b = 0; b < batch_size; b++) {
            float* row = ptr + b * num_classes;

            // Find max for numerical stability
            float max_val = row[0];
            for (size_t i = 1; i < num_classes; i++) {
                if (row[i] > max_val) max_val = row[i];
            }

            // Compute softmax
            float sum = 0.0f;
            for (size_t i = 0; i < num_classes; i++) {
                row[i] = std::exp(row[i] - max_val);
                sum += row[i];
            }
            for (size_t i = 0; i < num_classes; i++) {
                row[i] /= sum;
            }
        }

        return output;
    }
};

PYBIND11_MODULE(whitematter, m) {
    m.doc() = "Whitematter ML inference module";

    py::class_<ModelWrapper>(m, "Model")
        .def(py::init<>())
        .def("load", &ModelWrapper::load, "Load model from file",
             py::arg("path"), py::arg("arch") = "auto")
        .def("predict", &ModelWrapper::predict, "Run inference, returns logits",
             py::arg("input"))
        .def("predict_class", &ModelWrapper::predict_class, "Get predicted class index",
             py::arg("input"))
        .def("predict_proba", &ModelWrapper::predict_proba, "Get class probabilities",
             py::arg("input"))
        .def_readonly("loaded", &ModelWrapper::loaded)
        .def_readonly("model_type", &ModelWrapper::model_type);

    // CIFAR-10 class names
    m.def("cifar10_classes", []() {
        return std::vector<std::string>{
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };
    }, "Get CIFAR-10 class names");

    // MNIST class names
    m.def("mnist_classes", []() {
        return std::vector<std::string>{
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        };
    }, "Get MNIST class names");

    // CIFAR-10 normalization constants
    m.attr("CIFAR10_MEAN") = std::vector<float>{0.4914f, 0.4822f, 0.4465f};
    m.attr("CIFAR10_STD") = std::vector<float>{0.2470f, 0.2435f, 0.2616f};

    // MNIST normalization constants
    m.attr("MNIST_MEAN") = std::vector<float>{0.1307f};
    m.attr("MNIST_STD") = std::vector<float>{0.3081f};
}
