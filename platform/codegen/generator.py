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
    "step": "StepLR scheduler(&optimizer, {step_size}, {gamma}f);",
    "cosine": "CosineAnnealingLR scheduler(&optimizer, {T_max});",
    "exponential": "ExponentialLR scheduler(&optimizer, {gamma}f);",
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

        # Detect data type
        data_type = dataset_config.get("data_type", "image")

        # Generate layer code
        layers_code = self._generate_layers(architecture["layers"])

        # Generate optimizer code
        training = architecture.get("training", {})
        optimizer_code = self._generate_optimizer(training.get("optimizer", {}))
        scheduler_code = self._generate_scheduler(training.get("scheduler", {}))

        # Get training params
        epochs = training.get("epochs", 10)
        batch_size = training.get("batch_size", 64)

        # Generate full training code based on data type
        if data_type == "text":
            train_code = self._generate_text_training_code(
                layers_code=layers_code,
                optimizer_code=optimizer_code,
                scheduler_code=scheduler_code,
                epochs=epochs,
                batch_size=batch_size,
                dataset_config=dataset_config
            )
        else:
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

        # Generate inference code
        if data_type == "text":
            infer_code = self._generate_text_inference_code(
                layers_code=layers_code,
                dataset_config=dataset_config
            )
        else:
            infer_code = self._generate_inference_code(
                layers_code=layers_code,
                dataset_config=dataset_config
            )
        infer_cpp = output_dir / "infer.cpp"
        with open(infer_cpp, 'w') as f:
            f.write(infer_code)

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

        # Parameter aliases (allow different names for the same parameter)
        aliases = {
            "vocab_size": "num_embeddings",
            "embed_dim": "embedding_dim",
        }

        # Apply aliases
        aliased_params = {}
        for key, value in params.items():
            if key in aliases:
                aliased_params[aliases[key]] = value
            aliased_params[key] = value

        filled_params = {**defaults, **aliased_params}
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
        printf("Usage: %s <data_dir> <output_model> [resume_weights] [start_epoch]\\n", argv[0]);
        return 1;
    }}

    std::string data_dir = argv[1];
    std::string output_path = argv[2];
    std::string resume_weights = (argc > 3) ? argv[3] : "";
    int start_epoch = (argc > 4) ? std::atoi(argv[4]) : 0;

    // Disable stdout buffering for real-time output
    setbuf(stdout, NULL);

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

    // Load weights if resuming
    if (!resume_weights.empty()) {{
        printf("Resuming from weights: %s (epoch %d)\\n", resume_weights.c_str(), start_epoch);
        load_model(&model, resume_weights);
    }}

    CrossEntropyLoss criterion;
    {optimizer_code}
    {scheduler_include}

    CustomDataLoader train_loader(train_data, {batch_size}, true);
    CustomDataLoader test_loader(test_data, {batch_size}, false);

    // Count parameters
    size_t total_params = 0;
    for (const auto& p : model.parameters()) total_params += p->size();
    printf("Total parameters: %zu\\n\\n", total_params);

    printf("Training for {epochs} epochs (starting from %d)...\\n", start_epoch);
    printf("------------------------------------------------------------------\\n");

    float best_acc = 0.0f;

    for (int epoch = start_epoch; epoch < {epochs}; epoch++) {{
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

        // Save checkpoint after each epoch
        save_model(&model, output_path);
    }}

    printf("------------------------------------------------------------------\\n");
    printf("Training complete! Best accuracy: %.2f%%\\n\\n", best_acc);

    printf("Saving model to '%s'...\\n", output_path.c_str());
    save_model(&model, output_path);

    return 0;
}}
'''
        return code

    def _generate_inference_code(
        self,
        layers_code: str,
        dataset_config: dict
    ) -> str:
        """Generate inference-only code for the model."""
        num_classes = dataset_config.get("num_classes", 10)

        code = f'''// Auto-generated inference code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include "tensor.h"
#include "layer.h"
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

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        fprintf(stderr, "Usage: %s <model.bin> <input_tensor.bin>\\n", argv[0]);
        return 1;
    }}

    std::string model_path = argv[1];
    std::string input_path = argv[2];

    try {{
        // Build model with same architecture as training
        Sequential model({{
{layers_code}
        }});

        // Load weights
        if (!load_model(&model, model_path)) {{
            fprintf(stderr, "Failed to load model from: %s\\n", model_path.c_str());
            return 1;
        }}

        // Load input tensor
        auto input_file = load_tensor_file(input_path);
        auto input = Tensor::create(input_file.shape, false);
        input->data = input_file.data;

        // Run inference
        NoGradGuard no_grad;
        model.eval();
        auto output = model.forward(input);

        // Apply softmax for probabilities
        size_t num_classes = output->shape.back();
        std::vector<float> probs(num_classes);

        float max_val = output->data[0];
        for (size_t i = 1; i < num_classes; i++) {{
            max_val = std::max(max_val, output->data[i]);
        }}

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; i++) {{
            probs[i] = std::exp(output->data[i] - max_val);
            sum += probs[i];
        }}
        for (size_t i = 0; i < num_classes; i++) {{
            probs[i] /= sum;
        }}

        // Find predicted class
        size_t predicted = 0;
        float max_prob = probs[0];
        for (size_t i = 1; i < num_classes; i++) {{
            if (probs[i] > max_prob) {{
                max_prob = probs[i];
                predicted = i;
            }}
        }}

        // Output JSON result
        printf("{{\\"predicted_class\\": %zu, \\"probabilities\\": [", predicted);
        for (size_t i = 0; i < num_classes; i++) {{
            printf("%.6f%s", probs[i], i < num_classes - 1 ? ", " : "");
        }}
        printf("]}}\\n");

        return 0;
    }} catch (const std::exception& e) {{
        fprintf(stderr, "Error: %s\\n", e.what());
        return 1;
    }}
}}
'''
        return code

    def _generate_text_training_code(
        self,
        layers_code: str,
        optimizer_code: str,
        scheduler_code: str,
        epochs: int,
        batch_size: int,
        dataset_config: dict
    ) -> str:
        """Generate text model training code for language modeling using Transformer."""

        has_scheduler = bool(scheduler_code)
        scheduler_step = "scheduler.step();" if has_scheduler else ""
        scheduler_include = scheduler_code if has_scheduler else ""

        vocab_size = dataset_config.get("vocab_size", dataset_config.get("num_classes", 100))
        seq_length = dataset_config.get("seq_length", 128)

        # Extract layer parameters for the custom model
        embed_dim = 128
        num_heads = 4
        num_layers = 4
        ff_dim = 256
        dropout_rate = 0.1

        # Parse layers_code to extract dimensions
        import re
        if "embedding_dim" in layers_code:
            match = re.search(r'embedding_dim.*?(\d+)', layers_code)
            if match:
                embed_dim = int(match.group(1))
        if "hidden_size" in layers_code:
            match = re.search(r'hidden_size.*?(\d+)', layers_code)
            if match:
                embed_dim = int(match.group(1))  # Use as embed_dim for transformer
                ff_dim = embed_dim * 2
        if "num_heads" in layers_code:
            match = re.search(r'num_heads.*?(\d+)', layers_code)
            if match:
                num_heads = int(match.group(1))
        if "num_layers" in layers_code:
            match = re.search(r'num_layers.*?(\d+)', layers_code)
            if match:
                num_layers = int(match.group(1))

        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            num_heads = 4
            while embed_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        code = f'''// Auto-generated Transformer language model training code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
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

// Transformer Block: Self-Attention + FFN with residual connections
class TransformerBlock : public Module {{
public:
    MultiHeadAttention* attn;
    LayerNorm* ln1;
    LayerNorm* ln2;
    Linear* ff1;
    Linear* ff2;
    Dropout* dropout1;
    Dropout* dropout2;

    size_t embed_dim_;
    size_t ff_dim_;

    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_dim, float dropout = 0.1f)
        : embed_dim_(embed_dim), ff_dim_(ff_dim) {{
        attn = new MultiHeadAttention(embed_dim, num_heads);
        ln1 = new LayerNorm(embed_dim);
        ln2 = new LayerNorm(embed_dim);
        ff1 = new Linear(embed_dim, ff_dim);
        ff2 = new Linear(ff_dim, embed_dim);
        dropout1 = new Dropout(dropout);
        dropout2 = new Dropout(dropout);
    }}

    ~TransformerBlock() {{
        delete attn;
        delete ln1;
        delete ln2;
        delete ff1;
        delete ff2;
        delete dropout1;
        delete dropout2;
    }}

    // Forward with causal mask
    TensorPtr forward_with_mask(const TensorPtr& x, const TensorPtr& mask) {{
        // Self-attention with residual
        auto attn_out = attn->forward(x, x, x, mask);
        attn_out = dropout1->forward(attn_out);
        auto h = ln1->forward(x + attn_out);

        // FFN with residual: Linear -> ReLU -> Linear
        size_t batch = h->shape[0];
        size_t seq_len = h->shape[1];

        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto ff_out = ff1->forward(h_flat);
        ff_out = ff_out->relu();
        ff_out = ff2->forward(ff_out);
        ff_out = ff_out->reshape({{batch, seq_len, embed_dim_}});
        ff_out = dropout2->forward(ff_out);

        return ln2->forward(h + ff_out);
    }}

    TensorPtr forward(const TensorPtr& x) override {{
        // Create causal mask
        size_t seq_len = x->shape[1];
        auto mask = MultiHeadAttention::causal_mask(seq_len);
        return forward_with_mask(x, mask);
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : attn->parameters()) params.push_back(p);
        for (auto& p : ln1->parameters()) params.push_back(p);
        for (auto& p : ln2->parameters()) params.push_back(p);
        for (auto& p : ff1->parameters()) params.push_back(p);
        for (auto& p : ff2->parameters()) params.push_back(p);
        return params;
    }}

    void train() {{
        dropout1->train();
        dropout2->train();
    }}

    void eval() {{
        dropout1->eval();
        dropout2->eval();
    }}

    std::string name() const override {{ return "TransformerBlock"; }}
}};

// Transformer Language Model
class TransformerLM : public Module {{
public:
    Embedding* token_emb;
    TensorPtr pos_emb;  // Learned positional embeddings
    std::vector<TransformerBlock*> blocks;
    LayerNorm* ln_final;
    Linear* head;
    Dropout* dropout;

    size_t vocab_size_;
    size_t embed_dim_;
    size_t max_seq_len_;

    TransformerLM(size_t vocab_size, size_t embed_dim, size_t num_heads,
                  size_t num_layers, size_t ff_dim, size_t max_seq_len, float dropout_rate = 0.1f)
        : vocab_size_(vocab_size), embed_dim_(embed_dim), max_seq_len_(max_seq_len) {{

        token_emb = new Embedding(vocab_size, embed_dim);

        // Learned positional embeddings
        pos_emb = Tensor::randn({{max_seq_len, embed_dim}}, true);
        float scale = 0.02f;
        for (auto& v : pos_emb->data) v *= scale;

        // Transformer blocks
        for (size_t i = 0; i < num_layers; i++) {{
            blocks.push_back(new TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate));
        }}

        ln_final = new LayerNorm(embed_dim);
        head = new Linear(embed_dim, vocab_size);
        dropout = new Dropout(dropout_rate);
    }}

    ~TransformerLM() {{
        delete token_emb;
        for (auto* b : blocks) delete b;
        delete ln_final;
        delete head;
        delete dropout;
    }}

    // Forward: [batch, seq_len] -> [batch, seq_len, vocab_size]
    TensorPtr forward(const TensorPtr& x) override {{
        size_t batch = x->shape[0];
        size_t seq_len = x->shape[1];

        // Token embeddings: [batch, seq_len] -> [batch, seq_len, embed_dim]
        auto h = token_emb->forward(x);

        // Add positional embeddings (broadcast across batch)
        // pos_emb is [max_seq_len, embed_dim], we need [seq_len, embed_dim]
        for (size_t b = 0; b < batch; b++) {{
            for (size_t t = 0; t < seq_len; t++) {{
                for (size_t d = 0; d < embed_dim_; d++) {{
                    h->data[(b * seq_len + t) * embed_dim_ + d] += pos_emb->data[t * embed_dim_ + d];
                }}
            }}
        }}

        h = dropout->forward(h);

        // Create causal mask once
        auto mask = MultiHeadAttention::causal_mask(seq_len);

        // Pass through transformer blocks
        for (auto* block : blocks) {{
            h = block->forward_with_mask(h, mask);
        }}

        h = ln_final->forward(h);

        // Project to vocab: [batch * seq_len, embed_dim] -> [batch * seq_len, vocab_size]
        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto logits_flat = head->forward(h_flat);
        auto logits = logits_flat->reshape({{batch, seq_len, vocab_size_}});

        return logits;
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : token_emb->parameters()) params.push_back(p);
        params.push_back(pos_emb);  // Learnable positional embeddings
        for (auto* block : blocks) {{
            for (auto& p : block->parameters()) params.push_back(p);
        }}
        for (auto& p : ln_final->parameters()) params.push_back(p);
        for (auto& p : head->parameters()) params.push_back(p);
        return params;
    }}

    void train() {{
        dropout->train();
        for (auto* block : blocks) block->train();
    }}

    void eval() {{
        dropout->eval();
        for (auto* block : blocks) block->eval();
    }}

    std::string name() const override {{ return "TransformerLM"; }}
}};

// Alias for compatibility
using LanguageModel = TransformerLM;

struct TextDataset {{
    TensorPtr inputs;   // [num_sequences, seq_length]
    TensorPtr targets;  // [num_sequences, seq_length]
    size_t num_sequences;
    size_t seq_length;
}};

TextDataset load_text_dataset(const std::string& data_dir, bool train) {{
    std::string prefix = train ? "train" : "test";
    auto inputs_file = load_tensor_file(data_dir + "/" + prefix + "_inputs.bin");
    auto targets_file = load_tensor_file(data_dir + "/" + prefix + "_targets.bin");

    TextDataset ds;
    ds.inputs = Tensor::create(inputs_file.shape, false);
    ds.inputs->data = inputs_file.data;
    ds.targets = Tensor::create(targets_file.shape, false);
    ds.targets->data = targets_file.data;
    ds.num_sequences = inputs_file.shape[0];
    ds.seq_length = inputs_file.shape[1];
    return ds;
}}

class TextDataLoader {{
public:
    TextDataLoader(const TextDataset& dataset, size_t batch_size, bool shuffle)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_idx(0) {{
        indices.resize(dataset.num_sequences);
        for (size_t i = 0; i < dataset.num_sequences; i++) indices[i] = i;
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

    bool has_next() const {{ return current_idx < dataset.num_sequences; }}

    size_t num_batches() const {{
        return (dataset.num_sequences + batch_size - 1) / batch_size;
    }}

    std::pair<TensorPtr, TensorPtr> next_batch() {{
        size_t actual_batch = std::min(batch_size, dataset.num_sequences - current_idx);
        size_t seq_len = dataset.seq_length;

        // Create batch tensors [batch_size, seq_length]
        auto batch_inputs = Tensor::create({{actual_batch, seq_len}}, false);
        auto batch_targets = Tensor::create({{actual_batch, seq_len}}, false);

        for (size_t i = 0; i < actual_batch; i++) {{
            size_t idx = indices[current_idx + i];
            std::copy(
                dataset.inputs->data.begin() + idx * seq_len,
                dataset.inputs->data.begin() + (idx + 1) * seq_len,
                batch_inputs->data.begin() + i * seq_len
            );
            std::copy(
                dataset.targets->data.begin() + idx * seq_len,
                dataset.targets->data.begin() + (idx + 1) * seq_len,
                batch_targets->data.begin() + i * seq_len
            );
        }}

        current_idx += actual_batch;
        return {{batch_inputs, batch_targets}};
    }}

private:
    const TextDataset& dataset;
    size_t batch_size;
    bool shuffle;
    size_t current_idx;
    std::vector<size_t> indices;
}};

// Compute perplexity = exp(avg cross entropy loss)
float compute_perplexity(LanguageModel& model, TextDataLoader& loader, size_t vocab_size) {{
    NoGradGuard no_grad;
    model.eval();

    float total_loss = 0.0f;
    size_t total_tokens = 0;
    loader.reset();

    while (loader.has_next()) {{
        auto [inputs, targets] = loader.next_batch();
        auto output = model.forward(inputs);  // [batch, seq_len, vocab_size]

        size_t batch_size = inputs->shape[0];
        size_t seq_len = inputs->shape[1];

        // Compute cross-entropy loss for each position
        for (size_t b = 0; b < batch_size; b++) {{
            for (size_t t = 0; t < seq_len; t++) {{
                size_t target_idx = static_cast<size_t>(targets->data[b * seq_len + t]);
                if (target_idx >= vocab_size) continue;  // Skip padding

                // Softmax + log
                float max_val = output->data[(b * seq_len + t) * vocab_size];
                for (size_t v = 1; v < vocab_size; v++) {{
                    max_val = std::max(max_val, output->data[(b * seq_len + t) * vocab_size + v]);
                }}

                float sum_exp = 0.0f;
                for (size_t v = 0; v < vocab_size; v++) {{
                    sum_exp += std::exp(output->data[(b * seq_len + t) * vocab_size + v] - max_val);
                }}

                float log_prob = output->data[(b * seq_len + t) * vocab_size + target_idx] - max_val - std::log(sum_exp);
                total_loss -= log_prob;
                total_tokens++;
            }}
        }}
    }}

    model.train();
    float avg_loss = total_loss / total_tokens;
    return std::exp(avg_loss);
}}

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        printf("Usage: %s <data_dir> <output_model> [resume_weights] [start_epoch]\\n", argv[0]);
        return 1;
    }}

    std::string data_dir = argv[1];
    std::string output_path = argv[2];
    std::string resume_weights = (argc > 3) ? argv[3] : "";
    int start_epoch = (argc > 4) ? std::atoi(argv[4]) : 0;

    // Disable stdout buffering for real-time output
    setbuf(stdout, NULL);

    printf("Text Model Training (Language Model)\\n");
    printf("=====================================\\n\\n");

    printf("Loading dataset from '%s'...\\n", data_dir.c_str());
    auto train_data = load_text_dataset(data_dir, true);
    auto test_data = load_text_dataset(data_dir, false);

    printf("Train sequences: %zu\\n", train_data.num_sequences);
    printf("Test sequences: %zu\\n", test_data.num_sequences);
    printf("Sequence length: %zu\\n", train_data.seq_length);
    printf("Vocabulary size: {vocab_size}\\n");
    printf("Architecture: Transformer ({num_layers} layers, {num_heads} heads, dim={embed_dim})\\n\\n");

    // Build Transformer language model
    TransformerLM model({vocab_size}, {embed_dim}, {num_heads}, {num_layers}, {ff_dim}, train_data.seq_length, {dropout_rate}f);

    // Load weights if resuming
    if (!resume_weights.empty()) {{
        printf("Resuming from weights: %s (epoch %d)\\n", resume_weights.c_str(), start_epoch);
        load_model(&model, resume_weights);
    }}

    CrossEntropyLoss criterion;
    {optimizer_code}
    {scheduler_include}

    TextDataLoader train_loader(train_data, {batch_size}, true);
    TextDataLoader test_loader(test_data, {batch_size}, false);

    // Count parameters
    size_t total_params = 0;
    for (const auto& p : model.parameters()) total_params += p->size();
    printf("Total parameters: %zu\\n\\n", total_params);

    printf("Training for {epochs} epochs (starting from %d)...\\n", start_epoch);
    printf("------------------------------------------------------------------\\n");

    float best_ppl = 1e9f;

    for (int epoch = start_epoch; epoch < {epochs}; epoch++) {{
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;
        size_t total_tokens = 0;

        while (train_loader.has_next()) {{
            auto [inputs, targets] = train_loader.next_batch();

            optimizer.zero_grad();
            auto output = model.forward(inputs);

            // Reshape for cross-entropy: [batch * seq_len, vocab_size]
            size_t batch_size = inputs->shape[0];
            size_t seq_len = inputs->shape[1];
            size_t vocab_size = {vocab_size};

            // Use reshape to maintain gradient graph
            auto output_flat = output->reshape({{batch_size * seq_len, vocab_size}});

            auto targets_flat = targets->reshape({{batch_size * seq_len}});

            auto loss = criterion(output_flat, targets_flat);
            loss->backward();
            optimizer.step();

            total_loss += loss->data[0] * batch_size * seq_len;
            total_tokens += batch_size * seq_len;
            num_batches++;
        }}

        {scheduler_step}

        float avg_loss = total_loss / total_tokens;
        float train_ppl = std::exp(avg_loss);
        float test_ppl = compute_perplexity(model, test_loader, {vocab_size});
        best_ppl = std::min(best_ppl, test_ppl);

        // Output accuracy as inverse perplexity percentage for compatibility
        float acc = 100.0f / test_ppl;

        printf("Epoch %2d | Loss: %.4f | Test Acc: %.2f%% | PPL: %.2f\\n",
               epoch + 1, avg_loss, acc, test_ppl);

        // Save checkpoint after each epoch
        save_model(&model, output_path);
    }}

    printf("------------------------------------------------------------------\\n");
    printf("Training complete! Best perplexity: %.2f\\n\\n", best_ppl);

    printf("Saving model to '%s'...\\n", output_path.c_str());
    save_model(&model, output_path);

    return 0;
}}
'''
        return code

    def _generate_text_inference_code(
        self,
        layers_code: str,
        dataset_config: dict
    ) -> str:
        """Generate text generation inference code using Transformer."""
        vocab_size = dataset_config.get("vocab_size", dataset_config.get("num_classes", 100))
        seq_length = dataset_config.get("seq_length", 128)

        # Extract layer parameters (same as training)
        embed_dim = 128
        num_heads = 4
        num_layers = 4
        ff_dim = 256
        dropout_rate = 0.1

        import re
        if "embedding_dim" in layers_code:
            match = re.search(r'embedding_dim.*?(\d+)', layers_code)
            if match:
                embed_dim = int(match.group(1))
        if "hidden_size" in layers_code:
            match = re.search(r'hidden_size.*?(\d+)', layers_code)
            if match:
                embed_dim = int(match.group(1))
                ff_dim = embed_dim * 2
        if "num_heads" in layers_code:
            match = re.search(r'num_heads.*?(\d+)', layers_code)
            if match:
                num_heads = int(match.group(1))
        if "num_layers" in layers_code:
            match = re.search(r'num_layers.*?(\d+)', layers_code)
            if match:
                num_layers = int(match.group(1))

        if embed_dim % num_heads != 0:
            num_heads = 4
            while embed_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        code = f'''// Auto-generated Transformer text generation inference code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include "tensor.h"
#include "layer.h"
#include "serialize.h"

// Transformer Block (must match training architecture)
class TransformerBlock : public Module {{
public:
    MultiHeadAttention* attn;
    LayerNorm* ln1;
    LayerNorm* ln2;
    Linear* ff1;
    Linear* ff2;
    Dropout* dropout1;
    Dropout* dropout2;

    size_t embed_dim_;
    size_t ff_dim_;

    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_dim, float dropout = 0.1f)
        : embed_dim_(embed_dim), ff_dim_(ff_dim) {{
        attn = new MultiHeadAttention(embed_dim, num_heads);
        ln1 = new LayerNorm(embed_dim);
        ln2 = new LayerNorm(embed_dim);
        ff1 = new Linear(embed_dim, ff_dim);
        ff2 = new Linear(ff_dim, embed_dim);
        dropout1 = new Dropout(dropout);
        dropout2 = new Dropout(dropout);
    }}

    ~TransformerBlock() {{
        delete attn; delete ln1; delete ln2; delete ff1; delete ff2; delete dropout1; delete dropout2;
    }}

    TensorPtr forward_with_mask(const TensorPtr& x, const TensorPtr& mask) {{
        auto attn_out = attn->forward(x, x, x, mask);
        attn_out = dropout1->forward(attn_out);
        auto h = ln1->forward(x + attn_out);

        size_t batch = h->shape[0];
        size_t seq_len = h->shape[1];
        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto ff_out = ff1->forward(h_flat);
        ff_out = ff_out->relu();
        ff_out = ff2->forward(ff_out);
        ff_out = ff_out->reshape({{batch, seq_len, embed_dim_}});
        ff_out = dropout2->forward(ff_out);

        return ln2->forward(h + ff_out);
    }}

    TensorPtr forward(const TensorPtr& x) override {{
        auto mask = MultiHeadAttention::causal_mask(x->shape[1]);
        return forward_with_mask(x, mask);
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : attn->parameters()) params.push_back(p);
        for (auto& p : ln1->parameters()) params.push_back(p);
        for (auto& p : ln2->parameters()) params.push_back(p);
        for (auto& p : ff1->parameters()) params.push_back(p);
        for (auto& p : ff2->parameters()) params.push_back(p);
        return params;
    }}

    void train() {{ dropout1->train(); dropout2->train(); }}
    void eval() {{ dropout1->eval(); dropout2->eval(); }}
    std::string name() const override {{ return "TransformerBlock"; }}
}};

// Transformer Language Model
class TransformerLM : public Module {{
public:
    Embedding* token_emb;
    TensorPtr pos_emb;
    std::vector<TransformerBlock*> blocks;
    LayerNorm* ln_final;
    Linear* head;
    Dropout* dropout;

    size_t vocab_size_;
    size_t embed_dim_;
    size_t max_seq_len_;

    TransformerLM(size_t vocab_size, size_t embed_dim, size_t num_heads,
                  size_t num_layers, size_t ff_dim, size_t max_seq_len, float dropout_rate = 0.1f)
        : vocab_size_(vocab_size), embed_dim_(embed_dim), max_seq_len_(max_seq_len) {{

        token_emb = new Embedding(vocab_size, embed_dim);
        pos_emb = Tensor::randn({{max_seq_len, embed_dim}}, true);
        float scale = 0.02f;
        for (auto& v : pos_emb->data) v *= scale;

        for (size_t i = 0; i < num_layers; i++) {{
            blocks.push_back(new TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate));
        }}
        ln_final = new LayerNorm(embed_dim);
        head = new Linear(embed_dim, vocab_size);
        dropout = new Dropout(dropout_rate);
    }}

    ~TransformerLM() {{
        delete token_emb;
        for (auto* b : blocks) delete b;
        delete ln_final;
        delete head;
        delete dropout;
    }}

    TensorPtr forward(const TensorPtr& x) override {{
        size_t batch = x->shape[0];
        size_t seq_len = x->shape[1];

        auto h = token_emb->forward(x);

        for (size_t b = 0; b < batch; b++) {{
            for (size_t t = 0; t < seq_len; t++) {{
                for (size_t d = 0; d < embed_dim_; d++) {{
                    h->data[(b * seq_len + t) * embed_dim_ + d] += pos_emb->data[t * embed_dim_ + d];
                }}
            }}
        }}

        h = dropout->forward(h);
        auto mask = MultiHeadAttention::causal_mask(seq_len);

        for (auto* block : blocks) {{
            h = block->forward_with_mask(h, mask);
        }}

        h = ln_final->forward(h);
        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto logits_flat = head->forward(h_flat);
        return logits_flat->reshape({{batch, seq_len, vocab_size_}});
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : token_emb->parameters()) params.push_back(p);
        params.push_back(pos_emb);
        for (auto* block : blocks) {{
            for (auto& p : block->parameters()) params.push_back(p);
        }}
        for (auto& p : ln_final->parameters()) params.push_back(p);
        for (auto& p : head->parameters()) params.push_back(p);
        return params;
    }}

    void train() {{
        dropout->train();
        for (auto* block : blocks) block->train();
    }}

    void eval() {{
        dropout->eval();
        for (auto* block : blocks) block->eval();
    }}

    std::string name() const override {{ return "TransformerLM"; }}
}};

using LanguageModel = TransformerLM;

// Load vocabulary from JSON
#include <map>
std::map<int, std::string> load_vocabulary(const std::string& path) {{
    std::map<int, std::string> idx_to_token;
    std::ifstream f(path);
    if (!f) return idx_to_token;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Simple JSON parsing for idx_to_token
    size_t pos = content.find("\\"idx_to_token\\"");
    if (pos == std::string::npos) return idx_to_token;

    pos = content.find("{{", pos);
    size_t end = content.find("}}", pos);
    std::string inner = content.substr(pos + 1, end - pos - 1);

    // Parse "idx": "token" pairs
    size_t i = 0;
    while (i < inner.length()) {{
        size_t key_start = inner.find("\\"", i);
        if (key_start == std::string::npos) break;
        size_t key_end = inner.find("\\"", key_start + 1);
        std::string key = inner.substr(key_start + 1, key_end - key_start - 1);

        size_t val_start = inner.find("\\"", key_end + 1);
        if (val_start == std::string::npos) break;
        size_t val_end = inner.find("\\"", val_start + 1);
        std::string val = inner.substr(val_start + 1, val_end - val_start - 1);

        // Handle escape sequences
        std::string decoded;
        for (size_t j = 0; j < val.length(); j++) {{
            if (val[j] == '\\\\' && j + 1 < val.length()) {{
                if (val[j+1] == 'n') {{ decoded += '\\n'; j++; }}
                else if (val[j+1] == 't') {{ decoded += '\\t'; j++; }}
                else decoded += val[j];
            }} else decoded += val[j];
        }}

        idx_to_token[std::stoi(key)] = decoded;
        i = val_end + 1;
    }}

    return idx_to_token;
}}

std::map<std::string, int> load_token_to_idx(const std::string& path) {{
    auto idx_to_token = load_vocabulary(path);
    std::map<std::string, int> token_to_idx;
    for (auto& [idx, token] : idx_to_token) {{
        token_to_idx[token] = idx;
    }}
    return token_to_idx;
}}

int sample_token(const std::vector<float>& logits, float temperature, std::mt19937& rng) {{
    std::vector<float> probs(logits.size());
    float max_val = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {{
        probs[i] = std::exp((logits[i] - max_val) / temperature);
        sum += probs[i];
    }}
    for (size_t i = 0; i < logits.size(); i++) {{
        probs[i] /= sum;
    }}

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}}

int main(int argc, char* argv[]) {{
    if (argc < 4) {{
        fprintf(stderr, "Usage: %s <model.bin> <vocab.json> <prompt> [max_tokens] [temperature]\\n", argv[0]);
        return 1;
    }}

    std::string model_path = argv[1];
    std::string vocab_path = argv[2];
    std::string prompt = argv[3];
    int max_tokens = argc > 4 ? std::stoi(argv[4]) : 100;
    float temperature = argc > 5 ? std::stof(argv[5]) : 0.8f;

    try {{
        // Load vocabulary
        auto idx_to_token = load_vocabulary(vocab_path);
        auto token_to_idx = load_token_to_idx(vocab_path);

        if (idx_to_token.empty()) {{
            fprintf(stderr, "Failed to load vocabulary\\n");
            return 1;
        }}

        // Build Transformer language model (must match training)
        TransformerLM model({vocab_size}, {embed_dim}, {num_heads}, {num_layers}, {ff_dim}, {seq_length}, {dropout_rate}f);

        if (!load_model(&model, model_path)) {{
            fprintf(stderr, "Failed to load model\\n");
            return 1;
        }}

        NoGradGuard no_grad;
        model.eval();

        // Encode prompt (character-level)
        std::vector<int> tokens;
        for (char c : prompt) {{
            std::string s(1, c);
            if (token_to_idx.count(s)) {{
                tokens.push_back(token_to_idx[s]);
            }} else {{
                tokens.push_back(1); // <unk>
            }}
        }}

        std::random_device rd;
        std::mt19937 rng(rd());

        // Generate tokens
        std::string output = prompt;
        size_t seq_length = {seq_length};
        size_t vocab_size = {vocab_size};

        for (int i = 0; i < max_tokens; i++) {{
            // Prepare input: last seq_length tokens
            std::vector<float> input_data(seq_length, 0);  // padding
            size_t start = tokens.size() > seq_length ? tokens.size() - seq_length : 0;
            size_t len = std::min(tokens.size(), seq_length);
            for (size_t j = 0; j < len; j++) {{
                input_data[seq_length - len + j] = static_cast<float>(tokens[start + j]);
            }}

            auto input = Tensor::create({{1, seq_length}}, false);
            input->data = input_data;

            // Forward pass
            auto logits = model.forward(input);  // [1, seq_length, vocab_size]

            // Get logits for last position
            std::vector<float> last_logits(vocab_size);
            for (size_t v = 0; v < vocab_size; v++) {{
                last_logits[v] = logits->data[(seq_length - 1) * vocab_size + v];
            }}

            // Sample next token
            int next_token = sample_token(last_logits, temperature, rng);
            tokens.push_back(next_token);

            // Decode and output
            if (idx_to_token.count(next_token)) {{
                output += idx_to_token[next_token];
            }}

            // Stop on EOS
            if (next_token == 2) break;  // <eos>
        }}

        printf("%s\\n", output.c_str());
        return 0;

    }} catch (const std::exception& e) {{
        fprintf(stderr, "Error: %s\\n", e.what());
        return 1;
    }}
}}
'''
        return code

    def _generate_makefile(self) -> str:
        """Generate Makefile for compiling training and inference code."""
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

# Path to whitematter source (project root is two levels up from generated/{job_id}/)
PROJECT_ROOT = ../..
CORE_DIR = $(PROJECT_ROOT)/core
BUILD_DIR = $(PROJECT_ROOT)/build

OBJS = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/layer.o $(BUILD_DIR)/loss.o $(BUILD_DIR)/optimizer.o $(BUILD_DIR)/serialize.o

all: train infer

train: train.cpp $(OBJS)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

infer: infer.cpp $(OBJS)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

clean:
\trm -f train infer

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
