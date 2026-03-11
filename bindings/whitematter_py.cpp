#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "serialize.h"
#include <sstream>

namespace py = pybind11;

// --- Numpy <-> Tensor conversion helpers ---
static TensorPtr tensor_from_numpy(py::array_t<float> arr, bool requires_grad = false) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 0) {
        throw std::runtime_error("Tensor must have at least one dimension");
    }
    std::vector<size_t> shape;
    for (py::ssize_t i = 0; i < buf.ndim; i++) {
        shape.push_back(static_cast<size_t>(buf.shape[i]));
    }
    auto t = Tensor::create(shape, requires_grad);
    const float* ptr = static_cast<const float*>(buf.ptr);
    std::copy(ptr, ptr + t->size(), t->data());
    return t;
}

static TensorPtr tensor_from_numpy_int_labels(py::array arr, bool requires_grad = false) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Label array must be 1D");
    }
    size_t n = static_cast<size_t>(buf.shape[0]);
    auto t = Tensor::create({n}, requires_grad);
    float* out = t->data();
    if (buf.format == py::format_descriptor<float>::format()) {
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::copy(ptr, ptr + n, out);
    } else if (buf.format == py::format_descriptor<int>::format() ||
               buf.format == py::format_descriptor<int32_t>::format()) {
        const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
        for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(ptr[i]);
    } else if (buf.format == py::format_descriptor<int64_t>::format()) {
        const int64_t* ptr = static_cast<const int64_t*>(buf.ptr);
        for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(ptr[i]);
    } else {
        throw std::runtime_error("Label array must be float, int32, or int64");
    }
    return t;
}

static py::array_t<float> tensor_to_numpy(const TensorPtr& t) {
    std::vector<py::ssize_t> shape;
    for (size_t d : t->shape) shape.push_back(static_cast<py::ssize_t>(d));
    py::array_t<float> out(shape);
    py::buffer_info buf = out.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::copy(t->data(), t->data() + t->size(), ptr);
    return out;
}

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
        std::copy(ptr, ptr + tensor->size(), tensor->data());

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
        std::copy(output->data(), output->data() + output->size(), result_ptr);

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
    m.doc() = "Whitematter ML library: inference and training (Tensor, Sequential, Linear, Conv2d, optimizers, losses).";

    // ============== Tensor (training API) ==============
    py::class_<Tensor, TensorPtr>(m, "Tensor", "Differentiable tensor; supports backward() and numpy conversion.")
        .def(py::init([](py::array_t<float> arr, bool requires_grad) {
            return tensor_from_numpy(arr, requires_grad);
        }), py::arg("array"), py::arg("requires_grad") = false,
        "Create tensor from numpy array (float32).")
        .def_static("from_numpy", &tensor_from_numpy, py::arg("array"), py::arg("requires_grad") = false,
                    "Create tensor from numpy array (float32).")
        .def_static("create", [](const std::vector<size_t>& shape, bool requires_grad) {
            return Tensor::create(shape, requires_grad);
        }, py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("zeros", [](const std::vector<size_t>& shape, bool requires_grad) {
            return Tensor::zeros(shape, requires_grad);
        }, py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("ones", [](const std::vector<size_t>& shape, bool requires_grad) {
            return Tensor::ones(shape, requires_grad);
        }, py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("randn", [](const std::vector<size_t>& shape, bool requires_grad) {
            return Tensor::randn(shape, requires_grad);
        }, py::arg("shape"), py::arg("requires_grad") = false)
        .def("numpy", &tensor_to_numpy, "Return tensor as numpy array (float32).")
        .def("shape", [](const TensorPtr& t) {
            return t->shape;
        })
        .def("size", [](const TensorPtr& t) { return t->size(); })
        .def_property_readonly("requires_grad", [](const TensorPtr& t) { return t->requires_grad; })
        .def("backward", [](const TensorPtr& t) { t->backward(); })
        .def("zero_grad", [](const TensorPtr& t) { t->zero_grad(); })
        .def("item", [](const TensorPtr& t) { return t->item(); })
        .def("__repr__", [](const TensorPtr& t) {
            std::ostringstream os;
            os << "Tensor(shape=" << t->shape.size() << "D, size=" << t->size() << ")";
            return os.str();
        })
        // Operations needed for training
        .def("__add__", [](const TensorPtr& a, const TensorPtr& b) { return a->add(b); })
        .def("__sub__", [](const TensorPtr& a, const TensorPtr& b) { return a->sub(b); })
        .def("__mul__", [](const TensorPtr& a, const TensorPtr& b) { return a->mul(b); })
        .def("__mul__", [](const TensorPtr& a, float s) { return a->mul(s); })
        .def("__truediv__", [](const TensorPtr& a, const TensorPtr& b) { return a->div(b); })
        .def("matmul", [](const TensorPtr& a, const TensorPtr& b) { return a->matmul(b); })
        .def("relu", [](const TensorPtr& t) { return t->relu(); })
        .def("flatten", [](const TensorPtr& t, size_t start_dim) { return t->flatten(start_dim); }, py::arg("start_dim") = 1)
        .def("log_softmax", [](const TensorPtr& t, int dim) { return t->log_softmax(dim); }, py::arg("dim") = -1)
        .def("reshape", [](const TensorPtr& t, const std::vector<size_t>& shape) { return t->reshape(shape); });

    // ============== Grad mode (no_grad for inference) ==============
    py::class_<NoGradGuard>(m, "NoGradGuard", "Context guard that disables gradient tracking (e.g. for inference).")
        .def(py::init<>())
        .def("__enter__", [](py::object self) { return self; })
        .def("__exit__", [](NoGradGuard&, py::object, py::object, py::object) { return false; });
    m.def("no_grad", []() { return NoGradGuard(); }, "Return a context manager that disables gradient tracking.");
    m.def("grad_enabled", &GradMode::is_enabled, "Return True if gradient tracking is enabled.");
    m.def("set_grad_enabled", &GradMode::set_enabled, py::arg("enabled"), "Set gradient tracking on/off.");

    // ============== Module & Layers (use unique_ptr so layers can be transferred into Sequential) ==============
    py::class_<Module>(m, "Module", "Base class for all layers and models.")
        .def("forward", [](Module& m, const TensorPtr& x) { return m.forward(x); })
        .def("parameters", [](Module& m) { return m.parameters(); })
        .def("__call__", [](Module& m, const TensorPtr& x) { return m.forward(x); });

    py::class_<Sequential, Module, std::unique_ptr<Sequential>>(m, "Sequential", "Container that runs layers in sequence; use add() to build a model.")
        .def(py::init<>())
        .def("add", [](Sequential& s, std::unique_ptr<Module> mod) { s.add(mod.release()); }, py::arg("module"),
            "Add a layer (ownership is transferred; do not use the layer object after adding).")
        .def("forward", [](Sequential& s, const TensorPtr& x) { return s.forward(x); })
        .def("parameters", [](Sequential& s) { return s.parameters(); })
        .def("train", [](Sequential& s) { s.train(); })
        .def("eval", [](Sequential& s) { s.eval(); })
        .def("summary", [](const Sequential& s, py::object input_shape) {
            std::vector<size_t> shape;
            if (!input_shape.is_none()) {
                for (auto item : input_shape) shape.push_back(py::cast<size_t>(item));
            }
            s.summary(shape);
        }, py::arg("input_shape") = py::none(), "Print model summary (optional input_shape, e.g. [1,1,28,28] for MNIST).");

    py::class_<Linear, Module, std::unique_ptr<Linear>>(m, "Linear", "Fully connected layer: in_features -> out_features.")
        .def(py::init<size_t, size_t>(), py::arg("in_features"), py::arg("out_features"));

    py::class_<Conv2d, Module, std::unique_ptr<Conv2d>>(m, "Conv2d", "2D convolution.")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0);

    py::class_<ReLU, Module, std::unique_ptr<ReLU>>(m, "ReLU").def(py::init<>());
    py::class_<Flatten, Module, std::unique_ptr<Flatten>>(m, "Flatten").def(py::init<>());
    py::class_<Dropout, Module, std::unique_ptr<Dropout>>(m, "Dropout", "Dropout with probability p.")
        .def(py::init<float>(), py::arg("p") = 0.5f);
    py::class_<BatchNorm2d, Module, std::unique_ptr<BatchNorm2d>>(m, "BatchNorm2d")
        .def(py::init<size_t, float, float>(), py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f);
    py::class_<MaxPool2d, Module, std::unique_ptr<MaxPool2d>>(m, "MaxPool2d")
        .def(py::init<size_t, size_t>(), py::arg("kernel_size"), py::arg("stride") = 0);

    // ============== Optimizers ==============
    py::class_<Optimizer>(m, "Optimizer", "Base optimizer; use SGD or Adam.")
        .def("step", [](Optimizer& o) { o.step(); })
        .def("zero_grad", [](Optimizer& o) { o.zero_grad(); })
        .def_readwrite("lr", &Optimizer::lr);

    py::class_<SGD, Optimizer>(m, "SGD", "SGD optimizer (optional momentum).")
        .def(py::init<const std::vector<TensorPtr>&, float, float>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0f);

    py::class_<Adam, Optimizer>(m, "Adam", "Adam optimizer.")
        .def(py::init<const std::vector<TensorPtr>&, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f);

    // ============== Losses ==============
    py::class_<CrossEntropyLoss, std::unique_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss",
        "Cross-entropy loss for classification. prediction: [N, C], target: [N] class indices (int or float).")
        .def(py::init<>())
        .def("forward", [](CrossEntropyLoss& c, const TensorPtr& pred, const TensorPtr& target) { return c.forward(pred, target); })
        .def("__call__", [](CrossEntropyLoss& c, const TensorPtr& pred, const TensorPtr& target) { return c.forward(pred, target); });

    // Convenience: loss from numpy labels (target as int or float 1D array)
    m.def("cross_entropy", [](const TensorPtr& prediction, py::array labels) {
        TensorPtr target = tensor_from_numpy_int_labels(labels, false);
        CrossEntropyLoss crit;
        return crit.forward(prediction, target);
    }, py::arg("prediction"), py::arg("target"),
       "Cross-entropy loss. prediction: Tensor [N,C], target: 1D array of class indices (int or float).");

    // ============== Save / Load ==============
    m.def("save_model", &save_model, py::arg("module"), py::arg("path"), "Save model parameters to file.");
    m.def("load_model", &load_model, py::arg("module"), py::arg("path"), "Load model parameters from file.");

    // ============== Inference-only Model (backward compatible) ==============
    py::class_<ModelWrapper>(m, "Model",
        "Legacy inference-only model: load(path) then predict(input). For training, use Sequential, Tensor, Adam, CrossEntropyLoss.")
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
