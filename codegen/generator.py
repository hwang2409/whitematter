"""
Code Generator - generates C++ training code from architecture JSON.
Uses template-based generation for safety.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


# Whitelist of allowed layer types and their C++ constructors
LAYER_TEMPLATES = {
    "conv2d": "new Conv2d({in_channels}, {out_channels}, {kernel_size}, {stride}, {padding})",
    "batchnorm2d": "new BatchNorm2d({num_features})",
    "layernorm": "new LayerNorm({{{normalized_shape}}})",
    "linear": "new Linear({in_features}, {out_features})",
    "relu": "new ReLU()",
    "sigmoid": "new Sigmoid()",
    "tanh": "new Tanh()",
    "softmax": "new Softmax({dim})",
    "leakyrelu": "new LeakyReLU({negative_slope}f)",
    "maxpool2d": "new MaxPool2d({kernel_size})",
    "avgpool2d": "new AvgPool2d({kernel_size})",
    "dropout": "new Dropout({p}f)",
    "flatten": "new Flatten()",
    "embedding": "new Embedding({num_embeddings}, {embedding_dim})",
    "lstm": "new LSTM({input_size}, {hidden_size}, true)",
    "gru": "new GRU({input_size}, {hidden_size}, true)",
}

OPTIMIZER_TEMPLATES = {
    "sgd": "SGD optimizer(model.parameters(), {learning_rate}f, {momentum}f);",
    "adam": "Adam optimizer(model.parameters(), {learning_rate}f, {beta1}f, {beta2}f);",
}

SCHEDULER_TEMPLATES = {
    "none": "",
    "step": "StepLR scheduler(optimizer, {step_size}, {gamma}f);",
    "cosine": "CosineAnnealingLR scheduler(optimizer, {T_max});",
    "exponential": "ExponentialLR scheduler(optimizer, {gamma}f);",
}


class CodeGenerator:
    """Generate C++ training code from architecture specification."""

    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"

    def generate(
        self,
        architecture: Dict[str, Any],
        dataset_config: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        Generate training code from architecture and dataset config.

        Args:
            architecture: Architecture JSON with layers and training config
            dataset_config: Dataset preprocessing config (from image_processor)
            output_dir: Where to write generated files

        Returns:
            Path to generated train.cpp
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate layer code
        layers_code = self._generate_layers(architecture["layers"])

        # Generate optimizer code
        training = architecture.get("training", {})
        optimizer_code = self._generate_optimizer(training.get("optimizer", {}))
        scheduler_code = self._generate_scheduler(training.get("scheduler", {}))

        # Get training params
        epochs = training.get("epochs", 10)
        batch_size = training.get("batch_size", 64)

        # Generate full training code
        train_code = self._generate_training_code(
            layers_code=layers_code,
            optimizer_code=optimizer_code,
            scheduler_code=scheduler_code,
            epochs=epochs,
            batch_size=batch_size,
            dataset_config=dataset_config
        )

        # Write files
        train_cpp = output_dir / "train.cpp"
        with open(train_cpp, 'w') as f:
            f.write(train_code)

        # Generate Makefile
        makefile = output_dir / "Makefile"
        with open(makefile, 'w') as f:
            f.write(self._generate_makefile())

        # Copy architecture for reference
        with open(output_dir / "architecture.json", 'w') as f:
            json.dump(architecture, f, indent=2)

        return train_cpp

    def _generate_layers(self, layers: List[Dict[str, Any]]) -> str:
        """Generate C++ layer initialization code."""
        lines = []
        for layer in layers:
            layer_type = layer["type"].lower()
            params = layer.get("params", {})

            if layer_type not in LAYER_TEMPLATES:
                raise ValueError(f"Unknown layer type: {layer_type}")

            template = LAYER_TEMPLATES[layer_type]
            # Fill in parameters with defaults
            filled = self._fill_template(template, params, layer_type)
            lines.append(f"        {filled}")

        return ",\n".join(lines)

    def _fill_template(self, template: str, params: dict, layer_type: str) -> str:
        """Fill template with parameters, using defaults where needed."""
        # Default values for common parameters
        defaults = {
            "stride": 1,
            "padding": 0,
            "momentum": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
            "negative_slope": 0.01,
            "p": 0.5,
            "dim": -1,
        }

        filled_params = {**defaults, **params}
        try:
            return template.format(**filled_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter {e} for layer {layer_type}")

    def _generate_optimizer(self, optimizer_config: dict) -> str:
        """Generate optimizer initialization code."""
        opt_type = optimizer_config.get("type", "sgd").lower()
        params = optimizer_config.get("params", {})

        if opt_type not in OPTIMIZER_TEMPLATES:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        # Defaults
        defaults = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
        }

        filled_params = {**defaults, **params}
        return OPTIMIZER_TEMPLATES[opt_type].format(**filled_params)

    def _generate_scheduler(self, scheduler_config: dict) -> str:
        """Generate scheduler initialization code."""
        sched_type = scheduler_config.get("type", "none").lower()
        params = scheduler_config.get("params", {})

        if sched_type not in SCHEDULER_TEMPLATES:
            return ""

        if sched_type == "none":
            return ""

        defaults = {
            "step_size": 10,
            "gamma": 0.1,
            "T_max": 50,
        }

        filled_params = {**defaults, **params}
        return SCHEDULER_TEMPLATES[sched_type].format(**filled_params)

    def _generate_training_code(
        self,
        layers_code: str,
        optimizer_code: str,
        scheduler_code: str,
        epochs: int,
        batch_size: int,
        dataset_config: dict
    ) -> str:
        """Generate complete training code."""

        has_scheduler = bool(scheduler_code)
        scheduler_step = "scheduler.step();" if has_scheduler else ""
        scheduler_include = scheduler_code if has_scheduler else ""

        num_classes = dataset_config.get("num_classes", 10)
        input_shape = dataset_config.get("input_shape", [3, 32, 32])

        code = f'''// Auto-generated training code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "serialize.h"

// Binary tensor loading
struct TensorFile {{
    std::vector<size_t> shape;
    std::vector<float> data;
}};

TensorFile load_tensor_file(const std::string& path) {{
    TensorFile result;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x54454E53) throw std::runtime_error("Invalid tensor file");

    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);

    result.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; i++) {{
        uint64_t dim;
        f.read(reinterpret_cast<char*>(&dim), 8);
        result.shape[i] = dim;
    }}

    size_t total = 1;
    for (auto d : result.shape) total *= d;
    result.data.resize(total);
    f.read(reinterpret_cast<char*>(result.data.data()), total * sizeof(float));

    return result;
}}

struct CustomDataset {{
    TensorPtr images;
    TensorPtr labels;
    size_t num_samples;
}};

CustomDataset load_dataset(const std::string& data_dir, bool train) {{
    std::string prefix = train ? "train" : "test";
    auto images_file = load_tensor_file(data_dir + "/" + prefix + "_images.bin");
    auto labels_file = load_tensor_file(data_dir + "/" + prefix + "_labels.bin");

    CustomDataset ds;
    ds.images = Tensor::create(images_file.shape, false);
    ds.images->data = images_file.data;
    ds.labels = Tensor::create(labels_file.shape, false);
    ds.labels->data = labels_file.data;
    ds.num_samples = images_file.shape[0];
    return ds;
}}

class CustomDataLoader {{
public:
    CustomDataLoader(const CustomDataset& dataset, size_t batch_size, bool shuffle)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_idx(0) {{
        indices.resize(dataset.num_samples);
        for (size_t i = 0; i < dataset.num_samples; i++) indices[i] = i;
        if (shuffle) reset();
    }}

    void reset() {{
        current_idx = 0;
        if (shuffle) {{
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }}
    }}

    bool has_next() const {{ return current_idx < dataset.num_samples; }}

    size_t num_batches() const {{
        return (dataset.num_samples + batch_size - 1) / batch_size;
    }}

    std::pair<TensorPtr, TensorPtr> next_batch() {{
        size_t actual_batch = std::min(batch_size, dataset.num_samples - current_idx);

        // Get image shape (without batch dimension)
        std::vector<size_t> img_shape(dataset.images->shape.begin() + 1, dataset.images->shape.end());
        size_t img_size = 1;
        for (auto d : img_shape) img_size *= d;

        // Create batch tensors
        std::vector<size_t> batch_img_shape = {{actual_batch}};
        batch_img_shape.insert(batch_img_shape.end(), img_shape.begin(), img_shape.end());

        auto batch_images = Tensor::create(batch_img_shape, false);
        auto batch_labels = Tensor::create({{actual_batch}}, false);

        for (size_t i = 0; i < actual_batch; i++) {{
            size_t idx = indices[current_idx + i];
            std::copy(
                dataset.images->data.begin() + idx * img_size,
                dataset.images->data.begin() + (idx + 1) * img_size,
                batch_images->data.begin() + i * img_size
            );
            batch_labels->data[i] = dataset.labels->data[idx];
        }}

        current_idx += actual_batch;
        return {{batch_images, batch_labels}};
    }}

private:
    const CustomDataset& dataset;
    size_t batch_size;
    bool shuffle;
    size_t current_idx;
    std::vector<size_t> indices;
}};

float compute_accuracy(Sequential& model, CustomDataLoader& loader) {{
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0, total = 0;
    loader.reset();

    while (loader.has_next()) {{
        auto [images, labels] = loader.next_batch();
        auto output = model.forward(images);

        size_t batch_size = output->shape[0];
        size_t num_classes = output->shape[1];

        for (size_t i = 0; i < batch_size; i++) {{
            size_t predicted = 0;
            float max_val = output->data[i * num_classes];
            for (size_t j = 1; j < num_classes; j++) {{
                if (output->data[i * num_classes + j] > max_val) {{
                    max_val = output->data[i * num_classes + j];
                    predicted = j;
                }}
            }}
            if (predicted == static_cast<size_t>(labels->data[i])) correct++;
            total++;
        }}
    }}

    model.train();
    return static_cast<float>(correct) / total * 100.0f;
}}

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        printf("Usage: %s <data_dir> <output_model>\\n", argv[0]);
        return 1;
    }}

    std::string data_dir = argv[1];
    std::string output_path = argv[2];

    printf("Custom Model Training\\n");
    printf("=====================\\n\\n");

    printf("Loading dataset from '%s'...\\n", data_dir.c_str());
    auto train_data = load_dataset(data_dir, true);
    auto test_data = load_dataset(data_dir, false);

    printf("Train samples: %zu\\n", train_data.num_samples);
    printf("Test samples: %zu\\n\\n", test_data.num_samples);

    // Build model
    Sequential model({{
{layers_code}
    }});

    CrossEntropyLoss criterion;
    {optimizer_code}
    {scheduler_include}

    CustomDataLoader train_loader(train_data, {batch_size}, true);
    CustomDataLoader test_loader(test_data, {batch_size}, false);

    // Count parameters
    size_t total_params = 0;
    for (const auto& p : model.parameters()) total_params += p->size();
    printf("Total parameters: %zu\\n\\n", total_params);

    printf("Training for {epochs} epochs...\\n");
    printf("------------------------------------------------------------------\\n");

    float best_acc = 0.0f;

    for (int epoch = 0; epoch < {epochs}; epoch++) {{
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {{
            auto [images, labels] = train_loader.next_batch();

            optimizer.zero_grad();
            auto output = model.forward(images);
            auto loss = criterion(output, labels);
            loss->backward();
            optimizer.step();

            total_loss += loss->data[0];
            num_batches++;
        }}

        {scheduler_step}

        float avg_loss = total_loss / num_batches;
        float test_acc = compute_accuracy(model, test_loader);
        best_acc = std::max(best_acc, test_acc);

        printf("Epoch %2d | Loss: %.4f | Test Acc: %.2f%% | Best: %.2f%%\\n",
               epoch + 1, avg_loss, test_acc, best_acc);
    }}

    printf("------------------------------------------------------------------\\n");
    printf("Training complete! Best accuracy: %.2f%%\\n\\n", best_acc);

    printf("Saving model to '%s'...\\n", output_path.c_str());
    save_model(&model, output_path);

    return 0;
}}
'''
        return code

    def _generate_makefile(self) -> str:
        """Generate Makefile for compiling training code."""
        return '''# Auto-generated Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -ffast-math -funroll-loops
LDFLAGS =

# Detect macOS and add OpenMP flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -mcpu=apple-m1 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

# Path to whitematter source
WM_DIR = ../..

OBJS = $(WM_DIR)/tensor.o $(WM_DIR)/layer.o $(WM_DIR)/loss.o $(WM_DIR)/optimizer.o $(WM_DIR)/serialize.o

train: train.cpp $(OBJS)
\t$(CXX) $(CXXFLAGS) -I$(WM_DIR) -o $@ $^ $(LDFLAGS)

clean:
\trm -f train

.PHONY: clean
'''

    def validate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an architecture specification.

        Returns dict with 'valid', 'errors', 'warnings' keys.
        """
        errors = []
        warnings = []

        # Check required fields
        if "layers" not in architecture:
            errors.append("Missing 'layers' field")
        else:
            layers = architecture["layers"]
            if not layers:
                errors.append("No layers specified")
            else:
                # Validate each layer
                for i, layer in enumerate(layers):
                    if "type" not in layer:
                        errors.append(f"Layer {i}: missing 'type' field")
                    elif layer["type"].lower() not in LAYER_TEMPLATES:
                        errors.append(f"Layer {i}: unknown type '{layer['type']}'")

        # Check training config
        training = architecture.get("training", {})
        optimizer = training.get("optimizer", {})
        if optimizer.get("type", "sgd").lower() not in OPTIMIZER_TEMPLATES:
            errors.append(f"Unknown optimizer: {optimizer.get('type')}")

        # Parameter count warning
        # (simplified - real implementation would compute actual params)
        if len(architecture.get("layers", [])) > 50:
            warnings.append("Architecture has many layers, training may be slow")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
