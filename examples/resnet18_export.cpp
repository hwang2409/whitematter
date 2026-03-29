// ResNet-18 ONNX Export Tool
//
// Loads a trained ResNet-18 checkpoint (or freshly initialized weights) and
// exports all parameters plus BatchNorm running statistics to a single binary
// file.  A companion Python script (resnet18_to_onnx.py) reads this file and
// builds a valid ONNX model with the full ResNet-18 graph, including skip
// connections that the Sequential-only ONNX exporter cannot represent.
//
// Binary format:
//   magic:      uint32  (0x524E4554 = "RNET")
//   num_params: uint32
//   For each parameter:
//     name_len: uint32
//     name:     char[name_len]
//     ndim:     uint32
//     shape:    uint32[ndim]
//     data:     float[product(shape)]
//
// Usage:
//   ./build/resnet18_export <checkpoint.ckpt> [output.bin]
//   ./build/resnet18_export --no-checkpoint [output.bin]

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "tensor.h"
#include "layer.h"
#include "serialize.h"
#include "optimizer.h"

// ---------------------------------------------------------------------------
// ResNet-18 architecture (must match the training code exactly)
// ---------------------------------------------------------------------------
struct BasicBlock {
    Conv2d conv1, conv2;
    BatchNorm2d bn1, bn2;

    bool has_downsample;
    Conv2d down_conv;
    BatchNorm2d down_bn;

    BasicBlock(size_t in_ch, size_t out_ch, size_t stride = 1)
        : conv1(in_ch, out_ch, 3, stride, 1),
          conv2(out_ch, out_ch, 3, 1, 1),
          bn1(out_ch),
          bn2(out_ch),
          has_downsample(stride != 1 || in_ch != out_ch),
          down_conv(in_ch, out_ch, 1, stride, 0),
          down_bn(out_ch)
    {}

    TensorPtr forward(const TensorPtr& x) {
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = out->relu();
        out = conv2.forward(out);
        out = bn2.forward(out);

        TensorPtr shortcut;
        if (has_downsample) {
            shortcut = down_conv.forward(x);
            shortcut = down_bn.forward(shortcut);
        } else {
            shortcut = x;
        }

        out = out->add(shortcut);
        out = out->relu();
        return out;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto add = [&](std::vector<TensorPtr> p) {
            params.insert(params.end(), p.begin(), p.end());
        };
        add(conv1.parameters());
        add(bn1.parameters());
        add(conv2.parameters());
        add(bn2.parameters());
        if (has_downsample) {
            add(down_conv.parameters());
            add(down_bn.parameters());
        }
        return params;
    }

    void train() { bn1.train(); bn2.train(); if (has_downsample) down_bn.train(); }
    void eval()  { bn1.eval();  bn2.eval();  if (has_downsample) down_bn.eval();  }
};

struct ResNet18 {
    Conv2d conv1;
    BatchNorm2d bn1;

    BasicBlock layer1_0, layer1_1;
    BasicBlock layer2_0, layer2_1;
    BasicBlock layer3_0, layer3_1;
    BasicBlock layer4_0, layer4_1;

    AdaptiveAvgPool2d pool;
    Flatten flatten;
    Linear fc;

    ResNet18()
        : conv1(3, 64, 3, 1, 1),
          bn1(64),
          layer1_0(64, 64, 1),
          layer1_1(64, 64, 1),
          layer2_0(64, 128, 2),
          layer2_1(128, 128, 1),
          layer3_0(128, 256, 2),
          layer3_1(256, 256, 1),
          layer4_0(256, 512, 2),
          layer4_1(512, 512, 1),
          pool(1, 1),
          flatten(),
          fc(512, 10)
    {}

    TensorPtr forward(const TensorPtr& x) {
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = out->relu();
        out = layer1_0.forward(out);
        out = layer1_1.forward(out);
        out = layer2_0.forward(out);
        out = layer2_1.forward(out);
        out = layer3_0.forward(out);
        out = layer3_1.forward(out);
        out = layer4_0.forward(out);
        out = layer4_1.forward(out);
        out = pool.forward(out);
        out = flatten.forward(out);
        out = fc.forward(out);
        return out;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto add = [&](std::vector<TensorPtr> p) {
            params.insert(params.end(), p.begin(), p.end());
        };
        add(conv1.parameters());
        add(bn1.parameters());
        add(layer1_0.parameters());
        add(layer1_1.parameters());
        add(layer2_0.parameters());
        add(layer2_1.parameters());
        add(layer3_0.parameters());
        add(layer3_1.parameters());
        add(layer4_0.parameters());
        add(layer4_1.parameters());
        add(fc.parameters());
        return params;
    }

    void train() {
        bn1.train();
        layer1_0.train(); layer1_1.train();
        layer2_0.train(); layer2_1.train();
        layer3_0.train(); layer3_1.train();
        layer4_0.train(); layer4_1.train();
    }

    void eval() {
        bn1.eval();
        layer1_0.eval(); layer1_1.eval();
        layer2_0.eval(); layer2_1.eval();
        layer3_0.eval(); layer3_1.eval();
        layer4_0.eval(); layer4_1.eval();
    }
};

// ---------------------------------------------------------------------------
// Binary writer helpers
// ---------------------------------------------------------------------------
static bool write_u32(std::ofstream& out, uint32_t val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    return out.good();
}

static bool write_named_tensor(std::ofstream& out, const std::string& name,
                               const TensorPtr& tensor) {
    uint32_t name_len = static_cast<uint32_t>(name.size());
    if (!write_u32(out, name_len)) return false;
    out.write(name.c_str(), name_len);

    uint32_t ndim = static_cast<uint32_t>(tensor->shape.size());
    if (!write_u32(out, ndim)) return false;

    size_t numel = 1;
    for (auto d : tensor->shape) {
        if (!write_u32(out, static_cast<uint32_t>(d))) return false;
        numel *= d;
    }

    out.write(reinterpret_cast<const char*>(tensor->data()),
              static_cast<std::streamsize>(numel * sizeof(float)));
    return out.good();
}

// ---------------------------------------------------------------------------
// Collect all named tensors (parameters + running stats) from a BatchNorm2d.
// ---------------------------------------------------------------------------
static void collect_bn(const std::string& prefix, BatchNorm2d& bn,
                       std::vector<std::pair<std::string, TensorPtr>>& out) {
    out.push_back({prefix + ".weight", bn.gamma});
    out.push_back({prefix + ".bias", bn.beta});
    out.push_back({prefix + ".running_mean", bn.running_mean});
    out.push_back({prefix + ".running_var", bn.running_var});
}

static void collect_conv(const std::string& prefix, Conv2d& conv,
                         std::vector<std::pair<std::string, TensorPtr>>& out) {
    out.push_back({prefix + ".weight", conv.weight});
    out.push_back({prefix + ".bias", conv.bias});
}

static void collect_block(const std::string& prefix, BasicBlock& block,
                          std::vector<std::pair<std::string, TensorPtr>>& out) {
    collect_conv(prefix + ".conv1", block.conv1, out);
    collect_bn(prefix + ".bn1", block.bn1, out);
    collect_conv(prefix + ".conv2", block.conv2, out);
    collect_bn(prefix + ".bn2", block.bn2, out);
    if (block.has_downsample) {
        collect_conv(prefix + ".downsample.conv", block.down_conv, out);
        collect_bn(prefix + ".downsample.bn", block.down_bn, out);
    }
}

static std::vector<std::pair<std::string, TensorPtr>>
collect_all_tensors(ResNet18& model) {
    std::vector<std::pair<std::string, TensorPtr>> tensors;

    collect_conv("conv1", model.conv1, tensors);
    collect_bn("bn1", model.bn1, tensors);

    collect_block("layer1.0", model.layer1_0, tensors);
    collect_block("layer1.1", model.layer1_1, tensors);
    collect_block("layer2.0", model.layer2_0, tensors);
    collect_block("layer2.1", model.layer2_1, tensors);
    collect_block("layer3.0", model.layer3_0, tensors);
    collect_block("layer3.1", model.layer3_1, tensors);
    collect_block("layer4.0", model.layer4_0, tensors);
    collect_block("layer4.1", model.layer4_1, tensors);

    tensors.push_back({"fc.weight", model.fc.weight});
    tensors.push_back({"fc.bias", model.fc.bias});

    return tensors;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    printf("ResNet-18 ONNX Export Tool\n");
    printf("=========================\n\n");

    bool use_checkpoint = true;
    std::string ckpt_path;
    std::string output_path = "resnet18_weights.bin";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-checkpoint") {
            use_checkpoint = false;
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("Usage: %s [options] [checkpoint.ckpt] [output.bin]\n\n", argv[0]);
            printf("Options:\n");
            printf("  --no-checkpoint   Export with randomly initialized weights\n");
            printf("  --help, -h        Show this help message\n\n");
            printf("If no output path is given, writes to resnet18_weights.bin\n");
            printf("Then run:  python examples/resnet18_to_onnx.py resnet18_weights.bin resnet18.onnx\n");
            return 0;
        } else if (ckpt_path.empty() && use_checkpoint) {
            ckpt_path = argv[i];
        } else {
            output_path = argv[i];
        }
    }

    if (use_checkpoint && ckpt_path.empty()) {
        fprintf(stderr, "Error: No checkpoint path provided.\n");
        fprintf(stderr, "Usage: %s <checkpoint.ckpt> [output.bin]\n", argv[0]);
        fprintf(stderr, "       %s --no-checkpoint [output.bin]\n", argv[0]);
        return 1;
    }

    // Build model
    ResNet18 model;
    model.eval();

    // Load checkpoint if provided.
    // ResNet18 is not a Module subclass, so we cannot use load_model /
    // load_checkpoint directly.  Instead we replicate the logic: open the
    // file, read parameters in order, and copy into the model tensors.
    if (use_checkpoint) {
        printf("Loading checkpoint: %s\n", ckpt_path.c_str());

        std::ifstream in(ckpt_path, std::ios::binary);
        if (!in.is_open()) {
            fprintf(stderr, "Error: Cannot open '%s'\n", ckpt_path.c_str());
            return 1;
        }

        auto read_u32 = [&](uint32_t& v) -> bool {
            in.read(reinterpret_cast<char*>(&v), 4);
            return in.good();
        };
        auto read_f32 = [&](float& v) -> bool {
            in.read(reinterpret_cast<char*>(&v), 4);
            return in.good();
        };
        auto read_i32 = [&](int& v) -> bool {
            in.read(reinterpret_cast<char*>(&v), 4);
            return in.good();
        };

        uint32_t magic = 0;
        if (!read_u32(magic) || (magic != 0x574D4350 && magic != MODEL_MAGIC)) {
            fprintf(stderr, "Error: Invalid file magic in '%s' (got 0x%08X)\n", ckpt_path.c_str(), magic);
            return 1;
        }
        bool is_wmcp = (magic == 0x574D4350);  // checkpoint format from training

        auto params = model.parameters();
        uint32_t num_params = 0;

        if (is_wmcp) {
            // WMCP checkpoint: magic, epoch(i32), best_acc(f32), lr(f32), num_params(u32)
            int epoch = 0; float best_acc = 0, lr = 0;
            if (!read_i32(epoch) || !read_f32(best_acc) || !read_f32(lr)) {
                fprintf(stderr, "Error: Failed to read WMCP checkpoint header\n"); return 1;
            }
            if (!read_u32(num_params)) {
                fprintf(stderr, "Error: Failed to read param count\n"); return 1;
            }
            printf("  Checkpoint: epoch=%d, best_acc=%.2f%%, lr=%.6f\n",
                   epoch, best_acc, lr);
        } else {
            // RNET or plain format
            uint32_t version_or_nparams = 0;
            if (!read_u32(version_or_nparams)) {
                fprintf(stderr, "Error: Truncated file\n"); return 1;
            }
            if (version_or_nparams == 1) {
                int epoch = 0; float loss = 0, accuracy = 0;
                if (!read_i32(epoch) || !read_f32(loss) || !read_f32(accuracy)) {
                    fprintf(stderr, "Error: Failed to read checkpoint header\n"); return 1;
                }
                if (!read_u32(num_params)) {
                    fprintf(stderr, "Error: Failed to read param count\n"); return 1;
                }
            } else {
                num_params = version_or_nparams;
            }
        }

        if (num_params != static_cast<uint32_t>(params.size())) {
            fprintf(stderr, "Error: Parameter count mismatch. File has %u, model has %zu\n",
                    num_params, params.size());
            return 1;
        }

        // Read each parameter tensor
        for (size_t i = 0; i < num_params; i++) {
            uint32_t ndim = 0;
            if (!read_u32(ndim)) { fprintf(stderr, "Error reading param %zu\n", i); return 1; }

            std::vector<size_t> shape(ndim);
            size_t total_size = 1;
            for (uint32_t d = 0; d < ndim; d++) {
                uint64_t dim_val = 0;
                in.read(reinterpret_cast<char*>(&dim_val), sizeof(uint64_t));
                if (!in.good()) { fprintf(stderr, "Error reading param %zu shape\n", i); return 1; }
                shape[d] = (size_t)dim_val;
                total_size *= (size_t)dim_val;
            }

            if (shape != params[i]->shape) {
                fprintf(stderr, "Error: Shape mismatch for parameter %zu\n", i);
                return 1;
            }

            in.read(reinterpret_cast<char*>(params[i]->data()),
                    static_cast<std::streamsize>(total_size * sizeof(float)));
            if (!in.good()) {
                fprintf(stderr, "Error reading param %zu data\n", i);
                return 1;
            }
        }

        printf("  Loaded %u parameters from %s\n", num_params, ckpt_path.c_str());

        // Skip momentum buffers
        if (is_wmcp) {
            uint32_t nm = 0;
            in.read(reinterpret_cast<char*>(&nm), 4);
            for (uint32_t i = 0; i < nm && in.good(); i++) {
                uint64_t sz = 0;
                in.read(reinterpret_cast<char*>(&sz), 8);
                in.seekg(sz * sizeof(float), std::ios::cur);
            }

            // Load BatchNorm running stats
            uint32_t nbn = 0;
            if (in.read(reinterpret_cast<char*>(&nbn), 4) && nbn > 0) {
                // Collect all BN running stats from model in same order as save
                std::vector<TensorPtr> bn_stats;
                bn_stats.push_back(model.bn1.running_mean);
                bn_stats.push_back(model.bn1.running_var);
                auto add_block = [&](BasicBlock& blk) {
                    bn_stats.push_back(blk.bn1.running_mean);
                    bn_stats.push_back(blk.bn1.running_var);
                    bn_stats.push_back(blk.bn2.running_mean);
                    bn_stats.push_back(blk.bn2.running_var);
                    if (blk.has_downsample) {
                        bn_stats.push_back(blk.down_bn.running_mean);
                        bn_stats.push_back(blk.down_bn.running_var);
                    }
                };
                add_block(model.layer1_0); add_block(model.layer1_1);
                add_block(model.layer2_0); add_block(model.layer2_1);
                add_block(model.layer3_0); add_block(model.layer3_1);
                add_block(model.layer4_0); add_block(model.layer4_1);

                uint32_t loaded = std::min(nbn, (uint32_t)bn_stats.size());
                for (uint32_t i = 0; i < loaded; i++) {
                    uint64_t sz = 0;
                    in.read(reinterpret_cast<char*>(&sz), 8);
                    if (sz == bn_stats[i]->size()) {
                        in.read(reinterpret_cast<char*>(bn_stats[i]->data()), sz * sizeof(float));
                    } else {
                        in.seekg(sz * sizeof(float), std::ios::cur);
                    }
                }
                printf("  Loaded %u BatchNorm running stats\n", loaded);
            }
        }

        model.eval();
    } else {
        printf("Using randomly initialized weights (no checkpoint).\n");
    }

    // Collect all named tensors
    auto tensors = collect_all_tensors(model);

    printf("\nCollected %zu tensors:\n", tensors.size());
    size_t total_floats = 0;
    for (const auto& [name, tensor] : tensors) {
        printf("  %-40s [", name.c_str());
        for (size_t i = 0; i < tensor->shape.size(); i++) {
            printf("%zu%s", tensor->shape[i],
                   i < tensor->shape.size() - 1 ? ", " : "");
        }
        printf("]  (%zu floats)\n", tensor->size());
        total_floats += tensor->size();
    }
    printf("  Total: %zu parameters (%.2f MB)\n\n",
           total_floats, total_floats * 4.0f / (1024.0f * 1024.0f));

    // Write binary file
    constexpr uint32_t MAGIC = 0x524E4554;  // "RNET"

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Cannot open '%s' for writing\n", output_path.c_str());
        return 1;
    }

    if (!write_u32(out, MAGIC)) { fprintf(stderr, "Write error\n"); return 1; }
    if (!write_u32(out, static_cast<uint32_t>(tensors.size()))) {
        fprintf(stderr, "Write error\n"); return 1;
    }

    for (const auto& [name, tensor] : tensors) {
        if (!write_named_tensor(out, name, tensor)) {
            fprintf(stderr, "Error writing tensor '%s'\n", name.c_str());
            return 1;
        }
    }

    out.close();

    // Report file size
    std::ifstream check(output_path, std::ios::binary | std::ios::ate);
    size_t file_size = static_cast<size_t>(check.tellg());
    check.close();

    printf("Exported to: %s (%.2f MB)\n", output_path.c_str(),
           file_size / (1024.0f * 1024.0f));
    printf("\nNext step: convert to ONNX with:\n");
    printf("  python examples/resnet18_to_onnx.py %s resnet18.onnx\n", output_path.c_str());

    return 0;
}
