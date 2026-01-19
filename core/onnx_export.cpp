#include "onnx_export.h"
#include <fstream>
#include <cstdio>
#include <cstring>
#include <sstream>

// =============================================================================
// Protobuf wire format helpers (no protobuf dependency needed)
// =============================================================================

class ProtobufWriter {
public:
    std::vector<uint8_t> data;

    void write_varint(uint64_t value) {
        while (value > 0x7F) {
            data.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        data.push_back(static_cast<uint8_t>(value));
    }

    void write_tag(int field_number, int wire_type) {
        write_varint((field_number << 3) | wire_type);
    }

    void write_int64(int field_number, int64_t value) {
        write_tag(field_number, 0);  // wire type 0 = varint
        write_varint(static_cast<uint64_t>(value));
    }

    void write_string(int field_number, const std::string& value) {
        write_tag(field_number, 2);  // wire type 2 = length-delimited
        write_varint(value.size());
        data.insert(data.end(), value.begin(), value.end());
    }

    void write_bytes(int field_number, const std::vector<uint8_t>& value) {
        write_tag(field_number, 2);
        write_varint(value.size());
        data.insert(data.end(), value.begin(), value.end());
    }

    void write_bytes(int field_number, const uint8_t* ptr, size_t len) {
        write_tag(field_number, 2);
        write_varint(len);
        data.insert(data.end(), ptr, ptr + len);
    }

    void write_submessage(int field_number, const ProtobufWriter& sub) {
        write_tag(field_number, 2);
        write_varint(sub.data.size());
        data.insert(data.end(), sub.data.begin(), sub.data.end());
    }

    void write_float(int field_number, float value) {
        write_tag(field_number, 5);  // wire type 5 = 32-bit
        uint32_t bits;
        memcpy(&bits, &value, sizeof(float));
        data.push_back(bits & 0xFF);
        data.push_back((bits >> 8) & 0xFF);
        data.push_back((bits >> 16) & 0xFF);
        data.push_back((bits >> 24) & 0xFF);
    }

    void write_packed_int64(int field_number, const std::vector<int64_t>& values) {
        ProtobufWriter packed;
        for (auto v : values) {
            packed.write_varint(static_cast<uint64_t>(v));
        }
        write_bytes(field_number, packed.data);
    }

    void write_packed_floats(int field_number, const float* values, size_t count) {
        write_tag(field_number, 2);
        write_varint(count * sizeof(float));
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(values);
        data.insert(data.end(), ptr, ptr + count * sizeof(float));
    }
};

// =============================================================================
// ONNX protobuf message builders
// =============================================================================

// ONNX TensorProto data types
enum ONNXDataType {
    ONNX_FLOAT = 1,
    ONNX_INT64 = 7,
};

// Build TensorProto
ProtobufWriter build_tensor_proto(const std::string& name, const std::vector<int64_t>& dims,
                                   const float* data, size_t data_size) {
    ProtobufWriter tensor;
    // dims (field 1, repeated int64)
    tensor.write_packed_int64(1, dims);
    // data_type (field 2, int32) = FLOAT
    tensor.write_int64(2, ONNX_FLOAT);
    // float_data (field 4, repeated float) - packed
    tensor.write_packed_floats(4, data, data_size);
    // name (field 8, string)
    tensor.write_string(8, name);
    return tensor;
}

// Build TensorShapeProto
ProtobufWriter build_tensor_shape(const std::vector<int64_t>& dims) {
    ProtobufWriter shape;
    for (auto d : dims) {
        ProtobufWriter dim;
        dim.write_int64(1, d);  // dim_value
        shape.write_submessage(1, dim);  // dim (repeated)
    }
    return shape;
}

// Build TypeProto for tensor
ProtobufWriter build_type_proto(int elem_type, const std::vector<int64_t>& dims) {
    ProtobufWriter tensor_type;
    tensor_type.write_int64(1, elem_type);  // elem_type
    auto shape = build_tensor_shape(dims);
    tensor_type.write_submessage(2, shape);  // shape

    ProtobufWriter type_proto;
    type_proto.write_submessage(1, tensor_type);  // tensor_type
    return type_proto;
}

// Build ValueInfoProto
ProtobufWriter build_value_info(const std::string& name, const std::vector<int64_t>& dims) {
    ProtobufWriter value_info;
    value_info.write_string(1, name);  // name
    auto type_proto = build_type_proto(ONNX_FLOAT, dims);
    value_info.write_submessage(2, type_proto);  // type
    return value_info;
}

// Build AttributeProto
// ONNX AttributeProto fields: name=1, f=2, i=3, ints=8, type=20
// AttributeType enum: FLOAT=1, INT=2, INTS=7
ProtobufWriter build_attr_int(const std::string& name, int64_t value) {
    ProtobufWriter attr;
    attr.write_string(1, name);   // name (field 1)
    attr.write_int64(3, value);   // i (field 3)
    attr.write_int64(20, 2);      // type = INT (field 20, value 2)
    return attr;
}

ProtobufWriter build_attr_ints(const std::string& name, const std::vector<int64_t>& values) {
    ProtobufWriter attr;
    attr.write_string(1, name);          // name (field 1)
    for (auto v : values) {
        attr.write_int64(8, v);          // ints (field 8, repeated)
    }
    attr.write_int64(20, 7);             // type = INTS (field 20, value 7)
    return attr;
}

ProtobufWriter build_attr_float(const std::string& name, float value) {
    ProtobufWriter attr;
    attr.write_string(1, name);   // name (field 1)
    attr.write_float(2, value);   // f (field 2)
    attr.write_int64(20, 1);      // type = FLOAT (field 20, value 1)
    return attr;
}

// Build NodeProto
ProtobufWriter build_node(const std::string& op_type,
                          const std::vector<std::string>& inputs,
                          const std::vector<std::string>& outputs,
                          const std::string& name = "",
                          const std::vector<ProtobufWriter>& attrs = {}) {
    ProtobufWriter node;
    for (const auto& in : inputs) {
        node.write_string(1, in);  // input
    }
    for (const auto& out : outputs) {
        node.write_string(2, out);  // output
    }
    if (!name.empty()) {
        node.write_string(3, name);  // name
    }
    node.write_string(4, op_type);  // op_type
    for (const auto& attr : attrs) {
        node.write_submessage(5, attr);  // attribute
    }
    return node;
}

// Build OpsetIdProto
ProtobufWriter build_opset(const std::string& domain, int64_t version) {
    ProtobufWriter opset;
    if (!domain.empty()) {
        opset.write_string(1, domain);  // domain
    }
    opset.write_int64(2, version);  // version
    return opset;
}

// =============================================================================
// Layer-specific ONNX conversion
// =============================================================================

struct ONNXContext {
    std::vector<ProtobufWriter> nodes;
    std::vector<ProtobufWriter> initializers;
    std::vector<ProtobufWriter> value_infos;  // intermediate values
    std::string current_input;
    std::vector<int64_t> current_shape;
    int node_counter = 0;
    int weight_counter = 0;
    bool verbose = false;

    std::string new_name(const std::string& prefix) {
        return prefix + "_" + std::to_string(node_counter++);
    }

    std::string new_weight(const std::string& prefix) {
        return prefix + "_" + std::to_string(weight_counter++);
    }
};

bool convert_linear(Linear* layer, ONNXContext& ctx) {
    std::string weight_name = ctx.new_weight("linear_weight");
    std::string bias_name = ctx.new_weight("linear_bias");
    std::string output_name = ctx.new_name("linear_out");

    // Get weight and bias
    // Our Linear stores weight as [in_features, out_features]
    // forward: Y = X @ W + b  where X is [batch, in], W is [in, out]
    auto& W = layer->weight;
    auto& b = layer->bias;

    size_t in_features = W->shape[0];
    size_t out_features = W->shape[1];

    // Add weight initializer [in_features, out_features]
    std::vector<int64_t> weight_dims = {static_cast<int64_t>(in_features), static_cast<int64_t>(out_features)};
    ctx.initializers.push_back(build_tensor_proto(weight_name, weight_dims, W->data.data(), W->data.size()));

    // Add bias initializer [out_features]
    std::vector<int64_t> bias_dims = {static_cast<int64_t>(out_features)};
    ctx.initializers.push_back(build_tensor_proto(bias_name, bias_dims, b->data.data(), b->data.size()));

    // Create Gemm node: Y = alpha * A @ B + beta * C
    // A = input [batch, in], B = weight [in, out], C = bias [out]
    // No transpose needed since our weight is already [in, out]
    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_float("alpha", 1.0f));
    attrs.push_back(build_attr_float("beta", 1.0f));

    ctx.nodes.push_back(build_node("Gemm", {ctx.current_input, weight_name, bias_name}, {output_name}, "", attrs));

    // Update shape: batch dims + [out_features]
    ctx.current_shape.back() = static_cast<int64_t>(out_features);
    ctx.current_input = output_name;

    if (ctx.verbose) printf("  Linear(%zu, %zu) -> Gemm\n", in_features, out_features);
    return true;
}

bool convert_conv2d(Conv2d* layer, ONNXContext& ctx) {
    std::string weight_name = ctx.new_weight("conv_weight");
    std::string bias_name = ctx.new_weight("conv_bias");
    std::string output_name = ctx.new_name("conv_out");

    auto& W = layer->weight;
    auto& b = layer->bias;

    // Weight shape: [out_channels, in_channels, kH, kW]
    std::vector<int64_t> weight_dims;
    for (auto d : W->shape) weight_dims.push_back(static_cast<int64_t>(d));
    ctx.initializers.push_back(build_tensor_proto(weight_name, weight_dims, W->data.data(), W->data.size()));

    // Bias shape: [out_channels]
    std::vector<int64_t> bias_dims = {static_cast<int64_t>(b->shape[0])};
    ctx.initializers.push_back(build_tensor_proto(bias_name, bias_dims, b->data.data(), b->data.size()));

    // Conv attributes
    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_ints("kernel_shape", {static_cast<int64_t>(layer->kernel_size), static_cast<int64_t>(layer->kernel_size)}));
    attrs.push_back(build_attr_ints("strides", {static_cast<int64_t>(layer->stride), static_cast<int64_t>(layer->stride)}));
    attrs.push_back(build_attr_ints("pads", {static_cast<int64_t>(layer->padding), static_cast<int64_t>(layer->padding),
                                              static_cast<int64_t>(layer->padding), static_cast<int64_t>(layer->padding)}));

    ctx.nodes.push_back(build_node("Conv", {ctx.current_input, weight_name, bias_name}, {output_name}, "", attrs));

    // Update shape: [N, out_channels, H_out, W_out]
    int64_t out_channels = static_cast<int64_t>(layer->out_channels);
    int64_t H_in = ctx.current_shape[2];
    int64_t W_in = ctx.current_shape[3];
    int64_t H_out = (H_in + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    int64_t W_out = (W_in + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    ctx.current_shape = {ctx.current_shape[0], out_channels, H_out, W_out};
    ctx.current_input = output_name;

    if (ctx.verbose) printf("  Conv2d(%zu, %zu, %zu) -> Conv\n", layer->in_channels, layer->out_channels, layer->kernel_size);
    return true;
}

bool convert_relu(ONNXContext& ctx) {
    std::string output_name = ctx.new_name("relu_out");
    ctx.nodes.push_back(build_node("Relu", {ctx.current_input}, {output_name}));
    ctx.current_input = output_name;
    if (ctx.verbose) printf("  ReLU -> Relu\n");
    return true;
}

bool convert_sigmoid(ONNXContext& ctx) {
    std::string output_name = ctx.new_name("sigmoid_out");
    ctx.nodes.push_back(build_node("Sigmoid", {ctx.current_input}, {output_name}));
    ctx.current_input = output_name;
    if (ctx.verbose) printf("  Sigmoid -> Sigmoid\n");
    return true;
}

bool convert_tanh(ONNXContext& ctx) {
    std::string output_name = ctx.new_name("tanh_out");
    ctx.nodes.push_back(build_node("Tanh", {ctx.current_input}, {output_name}));
    ctx.current_input = output_name;
    if (ctx.verbose) printf("  Tanh -> Tanh\n");
    return true;
}

bool convert_softmax(Softmax* layer, ONNXContext& ctx) {
    std::string output_name = ctx.new_name("softmax_out");
    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_int("axis", layer->dim < 0 ? static_cast<int64_t>(ctx.current_shape.size()) + layer->dim : layer->dim));
    ctx.nodes.push_back(build_node("Softmax", {ctx.current_input}, {output_name}, "", attrs));
    ctx.current_input = output_name;
    if (ctx.verbose) printf("  Softmax(%d) -> Softmax\n", layer->dim);
    return true;
}

bool convert_flatten(ONNXContext& ctx) {
    std::string output_name = ctx.new_name("flatten_out");
    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_int("axis", 1));  // Flatten from dim 1 (keep batch)
    ctx.nodes.push_back(build_node("Flatten", {ctx.current_input}, {output_name}, "", attrs));

    // Compute flattened size
    int64_t flat_size = 1;
    for (size_t i = 1; i < ctx.current_shape.size(); i++) {
        flat_size *= ctx.current_shape[i];
    }
    ctx.current_shape = {ctx.current_shape[0], flat_size};
    ctx.current_input = output_name;

    if (ctx.verbose) printf("  Flatten -> Flatten (size=%lld)\n", flat_size);
    return true;
}

bool convert_maxpool2d(MaxPool2d* layer, ONNXContext& ctx) {
    std::string output_name = ctx.new_name("maxpool_out");

    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_ints("kernel_shape", {static_cast<int64_t>(layer->kernel_size), static_cast<int64_t>(layer->kernel_size)}));
    attrs.push_back(build_attr_ints("strides", {static_cast<int64_t>(layer->stride), static_cast<int64_t>(layer->stride)}));

    ctx.nodes.push_back(build_node("MaxPool", {ctx.current_input}, {output_name}, "", attrs));

    // Update shape
    int64_t H_out = ctx.current_shape[2] / static_cast<int64_t>(layer->stride);
    int64_t W_out = ctx.current_shape[3] / static_cast<int64_t>(layer->stride);
    ctx.current_shape = {ctx.current_shape[0], ctx.current_shape[1], H_out, W_out};
    ctx.current_input = output_name;

    if (ctx.verbose) printf("  MaxPool2d(%zu) -> MaxPool\n", layer->kernel_size);
    return true;
}

bool convert_avgpool2d(AvgPool2d* layer, ONNXContext& ctx) {
    std::string output_name = ctx.new_name("avgpool_out");

    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_ints("kernel_shape", {static_cast<int64_t>(layer->kernel_size), static_cast<int64_t>(layer->kernel_size)}));
    attrs.push_back(build_attr_ints("strides", {static_cast<int64_t>(layer->stride), static_cast<int64_t>(layer->stride)}));

    ctx.nodes.push_back(build_node("AveragePool", {ctx.current_input}, {output_name}, "", attrs));

    int64_t H_out = ctx.current_shape[2] / static_cast<int64_t>(layer->stride);
    int64_t W_out = ctx.current_shape[3] / static_cast<int64_t>(layer->stride);
    ctx.current_shape = {ctx.current_shape[0], ctx.current_shape[1], H_out, W_out};
    ctx.current_input = output_name;

    if (ctx.verbose) printf("  AvgPool2d(%zu) -> AveragePool\n", layer->kernel_size);
    return true;
}

bool convert_batchnorm2d(BatchNorm2d* layer, ONNXContext& ctx) {
    std::string scale_name = ctx.new_weight("bn_scale");
    std::string bias_name = ctx.new_weight("bn_bias");
    std::string mean_name = ctx.new_weight("bn_mean");
    std::string var_name = ctx.new_weight("bn_var");
    std::string output_name = ctx.new_name("bn_out");

    int64_t num_features = static_cast<int64_t>(layer->num_features);
    std::vector<int64_t> dims = {num_features};

    ctx.initializers.push_back(build_tensor_proto(scale_name, dims, layer->gamma->data.data(), layer->gamma->data.size()));
    ctx.initializers.push_back(build_tensor_proto(bias_name, dims, layer->beta->data.data(), layer->beta->data.size()));
    ctx.initializers.push_back(build_tensor_proto(mean_name, dims, layer->running_mean->data.data(), layer->running_mean->data.size()));
    ctx.initializers.push_back(build_tensor_proto(var_name, dims, layer->running_var->data.data(), layer->running_var->data.size()));

    std::vector<ProtobufWriter> attrs;
    attrs.push_back(build_attr_float("epsilon", layer->eps));

    ctx.nodes.push_back(build_node("BatchNormalization",
        {ctx.current_input, scale_name, bias_name, mean_name, var_name},
        {output_name}, "", attrs));
    ctx.current_input = output_name;

    if (ctx.verbose) printf("  BatchNorm2d(%zu) -> BatchNormalization\n", layer->num_features);
    return true;
}

bool convert_dropout(ONNXContext& ctx) {
    // In inference mode, dropout is identity - we can skip it or add Identity node
    std::string output_name = ctx.new_name("dropout_out");
    ctx.nodes.push_back(build_node("Identity", {ctx.current_input}, {output_name}));
    ctx.current_input = output_name;
    if (ctx.verbose) printf("  Dropout -> Identity (inference mode)\n");
    return true;
}

// =============================================================================
// Main export function
// =============================================================================

bool export_onnx(Sequential* model, const std::string& filepath, const ONNXExportOptions& options) {
    if (options.input_shape.empty()) {
        fprintf(stderr, "ONNX export error: input_shape is required\n");
        return false;
    }

    ONNXContext ctx;
    ctx.verbose = options.verbose;
    ctx.current_input = "input";

    // Convert input shape to int64
    for (auto d : options.input_shape) {
        ctx.current_shape.push_back(static_cast<int64_t>(d));
    }

    if (options.verbose) {
        printf("ONNX Export: %s\n", options.model_name.c_str());
        printf("Input shape: [");
        for (size_t i = 0; i < ctx.current_shape.size(); i++) {
            printf("%lld%s", ctx.current_shape[i], i < ctx.current_shape.size() - 1 ? ", " : "");
        }
        printf("]\n");
        printf("Converting layers:\n");
    }

    // Convert each layer
    for (Module* layer : model->layers) {
        if (auto* l = dynamic_cast<Linear*>(layer)) {
            if (!convert_linear(l, ctx)) return false;
        } else if (auto* l = dynamic_cast<Conv2d*>(layer)) {
            if (!convert_conv2d(l, ctx)) return false;
        } else if (dynamic_cast<ReLU*>(layer)) {
            if (!convert_relu(ctx)) return false;
        } else if (dynamic_cast<Sigmoid*>(layer)) {
            if (!convert_sigmoid(ctx)) return false;
        } else if (dynamic_cast<Tanh*>(layer)) {
            if (!convert_tanh(ctx)) return false;
        } else if (auto* l = dynamic_cast<Softmax*>(layer)) {
            if (!convert_softmax(l, ctx)) return false;
        } else if (dynamic_cast<Flatten*>(layer)) {
            if (!convert_flatten(ctx)) return false;
        } else if (auto* l = dynamic_cast<MaxPool2d*>(layer)) {
            if (!convert_maxpool2d(l, ctx)) return false;
        } else if (auto* l = dynamic_cast<AvgPool2d*>(layer)) {
            if (!convert_avgpool2d(l, ctx)) return false;
        } else if (auto* l = dynamic_cast<BatchNorm2d*>(layer)) {
            if (!convert_batchnorm2d(l, ctx)) return false;
        } else if (dynamic_cast<Dropout*>(layer)) {
            if (!convert_dropout(ctx)) return false;
        } else {
            fprintf(stderr, "ONNX export error: Unsupported layer type\n");
            return false;
        }
    }

    // Build graph
    ProtobufWriter graph;

    // Add nodes
    for (const auto& node : ctx.nodes) {
        graph.write_submessage(1, node);  // node (field 1)
    }

    // Add graph name
    graph.write_string(2, options.model_name);  // name (field 2)

    // Add initializers (weights)
    for (const auto& init : ctx.initializers) {
        graph.write_submessage(5, init);  // initializer (field 5)
    }

    // Add input
    std::vector<int64_t> input_dims;
    for (auto d : options.input_shape) input_dims.push_back(static_cast<int64_t>(d));
    graph.write_submessage(11, build_value_info("input", input_dims));  // input (field 11)

    // Add output
    graph.write_submessage(12, build_value_info(ctx.current_input, ctx.current_shape));  // output (field 12)

    // Build model
    ProtobufWriter model_proto;
    model_proto.write_int64(1, 8);  // ir_version (field 1) = 8

    // Opset import
    model_proto.write_submessage(8, build_opset("", ONNX_OPSET_VERSION));  // opset_import (field 8)

    model_proto.write_string(2, options.producer_name);     // producer_name (field 2)
    model_proto.write_string(3, options.producer_version);  // producer_version (field 3)
    model_proto.write_submessage(7, graph);                 // graph (field 7)

    // Write to file
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        fprintf(stderr, "ONNX export error: Cannot open file '%s'\n", filepath.c_str());
        return false;
    }
    file.write(reinterpret_cast<const char*>(model_proto.data.data()), model_proto.data.size());
    file.close();

    if (options.verbose) {
        printf("Output shape: [");
        for (size_t i = 0; i < ctx.current_shape.size(); i++) {
            printf("%lld%s", ctx.current_shape[i], i < ctx.current_shape.size() - 1 ? ", " : "");
        }
        printf("]\n");
        printf("Exported %zu nodes, %zu initializers\n", ctx.nodes.size(), ctx.initializers.size());
        printf("Saved to: %s (%zu bytes)\n", filepath.c_str(), model_proto.data.size());
    }

    return true;
}

bool export_onnx(Sequential* model, const std::string& filepath, const std::vector<size_t>& input_shape) {
    ONNXExportOptions options;
    options.input_shape = input_shape;
    return export_onnx(model, filepath, options);
}

std::string get_onnx_export_info(Sequential* model, const std::vector<size_t>& input_shape) {
    std::stringstream ss;
    ss << "ONNX Export Info\n";
    ss << "================\n";
    ss << "Input shape: [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        ss << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
    }
    ss << "]\n\n";
    ss << "Layers:\n";

    for (Module* layer : model->layers) {
        if (auto* l = dynamic_cast<Linear*>(layer)) {
            ss << "  Linear(" << l->weight->shape[0] << ", " << l->weight->shape[1] << ") -> Gemm\n";
        } else if (auto* l = dynamic_cast<Conv2d*>(layer)) {
            ss << "  Conv2d(" << l->in_channels << ", " << l->out_channels << ", " << l->kernel_size << ") -> Conv\n";
        } else if (dynamic_cast<ReLU*>(layer)) {
            ss << "  ReLU -> Relu\n";
        } else if (dynamic_cast<Sigmoid*>(layer)) {
            ss << "  Sigmoid -> Sigmoid\n";
        } else if (dynamic_cast<Tanh*>(layer)) {
            ss << "  Tanh -> Tanh\n";
        } else if (auto* l = dynamic_cast<Softmax*>(layer)) {
            ss << "  Softmax(" << l->dim << ") -> Softmax\n";
        } else if (dynamic_cast<Flatten*>(layer)) {
            ss << "  Flatten -> Flatten\n";
        } else if (auto* l = dynamic_cast<MaxPool2d*>(layer)) {
            ss << "  MaxPool2d(" << l->kernel_size << ") -> MaxPool\n";
        } else if (auto* l = dynamic_cast<AvgPool2d*>(layer)) {
            ss << "  AvgPool2d(" << l->kernel_size << ") -> AveragePool\n";
        } else if (auto* l = dynamic_cast<BatchNorm2d*>(layer)) {
            ss << "  BatchNorm2d(" << l->num_features << ") -> BatchNormalization\n";
        } else if (dynamic_cast<Dropout*>(layer)) {
            ss << "  Dropout -> Identity\n";
        } else {
            ss << "  [Unsupported layer]\n";
        }
    }

    return ss.str();
}
