#include "../onnx_import.h"
#include "../onnx_export.h"
#include <fstream>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

// =============================================================================
// Protobuf wire format reader (mirrors export; no protobuf dependency)
// =============================================================================
// Wire types: 0=varint, 1=64-bit, 2=length-delimited, 5=32-bit

class ProtobufReader {
public:
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    void reset(const uint8_t* d, size_t s) {
        data = d;
        size = s;
        pos = 0;
    }

    bool eof() const { return pos >= size; }

    int read_tag(int& field_number, int& wire_type) {
        if (pos >= size) return -1;
        uint64_t tag;
        if (!read_varint(tag)) return -1;
        wire_type = static_cast<int>(tag & 7);
        field_number = static_cast<int>(tag >> 3);
        return 0;
    }

    bool read_varint(uint64_t& out) {
        out = 0;
        int shift = 0;
        while (pos < size) {
            uint8_t b = data[pos++];
            out |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return true;
            shift += 7;
            if (shift >= 64) return false;
        }
        return false;
    }

    bool read_length_delimited(std::vector<uint8_t>& out) {
        uint64_t len;
        if (!read_varint(len) || pos + len > size) return false;
        out.assign(data + pos, data + pos + len);
        pos += static_cast<size_t>(len);
        return true;
    }

    bool read_fixed32(uint32_t& out) {
        if (pos + 4 > size) return false;
        out = static_cast<uint32_t>(data[pos]) |
              (static_cast<uint32_t>(data[pos + 1]) << 8) |
              (static_cast<uint32_t>(data[pos + 2]) << 16) |
              (static_cast<uint32_t>(data[pos + 3]) << 24);
        pos += 4;
        return true;
    }

    bool skip_field(int wire_type) {
        if (wire_type == 0) {
            uint64_t v;
            return read_varint(v);
        }
        if (wire_type == 1) {
            if (pos + 8 > size) return false;
            pos += 8;
            return true;
        }
        if (wire_type == 2) {
            uint64_t len;
            if (!read_varint(len) || pos + len > size) return false;
            pos += static_cast<size_t>(len);
            return true;
        }
        if (wire_type == 5) {
            if (pos + 4 > size) return false;
            pos += 4;
            return true;
        }
        return false;
    }
};

// =============================================================================
// ONNX structures (minimal for our export format)
// =============================================================================

struct TensorInit {
    std::string name;
    std::vector<int64_t> dims;
    std::vector<float> float_data;
    int data_type = 1;  // FLOAT
};

struct NodeInfo {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, int64_t> attrs_i;
    std::unordered_map<std::string, float> attrs_f;
    std::unordered_map<std::string, std::vector<int64_t>> attrs_ints;
};

// Parse TensorProto from a length-delimited buffer
bool parse_tensor_proto(const uint8_t* buf, size_t len, TensorInit& out) {
    ProtobufReader r;
    r.reset(buf, len);
    out.dims.clear();
    out.float_data.clear();
    out.data_type = 1;

    while (!r.eof()) {
        int fn, wt;
        if (r.read_tag(fn, wt) != 0) break;
        if (fn == 1) {  // dims (repeated int64)
            std::vector<uint8_t> packed;
            if (!r.read_length_delimited(packed)) return false;
            ProtobufReader pr;
            pr.reset(packed.data(), packed.size());
            while (!pr.eof()) {
                uint64_t v;
                if (!pr.read_varint(v)) break;
                out.dims.push_back(static_cast<int64_t>(v));
            }
        } else if (fn == 2) {  // data_type
            uint64_t v;
            if (!r.read_varint(v)) return false;
            out.data_type = static_cast<int>(v);
        } else if (fn == 4) {  // float_data (packed float)
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            size_t n = raw.size() / sizeof(float);
            out.float_data.resize(n);
            if (n) memcpy(out.float_data.data(), raw.data(), n * sizeof(float));
        } else if (fn == 8) {  // name
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            out.name.assign(raw.begin(), raw.end());
        } else {
            if (!r.skip_field(wt)) return false;
        }
    }
    return true;
}

// Parse ValueInfoProto to get tensor shape (for input)
bool parse_value_info_shape(const uint8_t* buf, size_t len, std::vector<int64_t>& dims) {
    dims.clear();
    ProtobufReader r;
    r.reset(buf, len);
    while (!r.eof()) {
        int fn, wt;
        if (r.read_tag(fn, wt) != 0) break;
        if (fn == 2) {  // type (TypeProto)
            std::vector<uint8_t> type_buf;
            if (!r.read_length_delimited(type_buf)) return false;
            ProtobufReader r2;
            r2.reset(type_buf.data(), type_buf.size());
            while (!r2.eof()) {
                int fn2, wt2;
                if (r2.read_tag(fn2, wt2) != 0) break;
                if (fn2 == 1) {  // tensor_type
                    std::vector<uint8_t> tt_buf;
                    if (!r2.read_length_delimited(tt_buf)) return false;
                    ProtobufReader r3;
                    r3.reset(tt_buf.data(), tt_buf.size());
                    while (!r3.eof()) {
                        int fn3, wt3;
                        if (r3.read_tag(fn3, wt3) != 0) break;
                        if (fn3 == 2) {  // shape
                            std::vector<uint8_t> shape_buf;
                            if (!r3.read_length_delimited(shape_buf)) return false;
                            ProtobufReader r4;
                            r4.reset(shape_buf.data(), shape_buf.size());
                            while (!r4.eof()) {
                                int fn4, wt4;
                                if (r4.read_tag(fn4, wt4) != 0) break;
                                if (fn4 == 1) {  // dim (repeated)
                                    std::vector<uint8_t> dim_buf;
                                    if (!r4.read_length_delimited(dim_buf)) return false;
                                    ProtobufReader r5;
                                    r5.reset(dim_buf.data(), dim_buf.size());
                                    while (!r5.eof()) {
                                        int fn5, wt5;
                                        if (r5.read_tag(fn5, wt5) != 0) break;
                                        if (fn5 == 1) {
                                            uint64_t v;
                                            if (r5.read_varint(v)) dims.push_back(static_cast<int64_t>(v));
                                        } else {
                                            r5.skip_field(wt5);
                                        }
                                    }
                                } else {
                                    r4.skip_field(wt4);
                                }
                            }
                        } else {
                            r3.skip_field(wt3);
                        }
                    }
                } else {
                    r2.skip_field(wt2);
                }
            }
        } else {
            r.skip_field(wt);
        }
    }
    return true;
}

// Parse NodeProto
bool parse_node_proto(const uint8_t* buf, size_t len, NodeInfo& out) {
    out.op_type.clear();
    out.inputs.clear();
    out.outputs.clear();
    out.attrs_i.clear();
    out.attrs_f.clear();
    out.attrs_ints.clear();
    ProtobufReader r;
    r.reset(buf, len);
    while (!r.eof()) {
        int fn, wt;
        if (r.read_tag(fn, wt) != 0) break;
        if (fn == 1) {  // input (repeated string)
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            out.inputs.emplace_back(raw.begin(), raw.end());
        } else if (fn == 2) {  // output
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            out.outputs.emplace_back(raw.begin(), raw.end());
        } else if (fn == 4) {  // op_type
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            out.op_type.assign(raw.begin(), raw.end());
        } else if (fn == 5) {  // attribute (repeated AttributeProto)
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            ProtobufReader ar;
            ar.reset(raw.data(), raw.size());
            std::string attr_name;
            while (!ar.eof()) {
                int afn, awt;
                if (ar.read_tag(afn, awt) != 0) break;
                if (afn == 1) {
                    std::vector<uint8_t> n;
                    if (!ar.read_length_delimited(n)) break;
                    attr_name.assign(n.begin(), n.end());
                } else if (afn == 2) {
                    uint32_t bits;
                    if (!ar.read_fixed32(bits)) break;
                    float f;
                    memcpy(&f, &bits, sizeof(float));
                    out.attrs_f[attr_name] = f;
                } else if (afn == 3) {
                    uint64_t v;
                    if (!ar.read_varint(v)) break;
                    out.attrs_i[attr_name] = static_cast<int64_t>(v);
                } else if (afn == 8) {
                    uint64_t v;
                    if (ar.read_varint(v)) out.attrs_ints[attr_name].push_back(static_cast<int64_t>(v));
                } else if (afn == 20) {
                    ar.skip_field(awt);  // type
                } else {
                    ar.skip_field(awt);
                }
            }
        } else {
            r.skip_field(wt);
        }
    }
    return true;
}

// Parse GraphProto: collect initializers, input shape, and nodes in order
struct GraphState {
    std::unordered_map<std::string, TensorInit> initializers;
    std::vector<int64_t> input_shape;
    std::vector<NodeInfo> nodes;
};

bool parse_graph_proto(const uint8_t* buf, size_t len, GraphState& g) {
    ProtobufReader r;
    r.reset(buf, len);
    while (!r.eof()) {
        int fn, wt;
        if (r.read_tag(fn, wt) != 0) break;
        if (fn == 1) {  // node (repeated NodeProto)
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            NodeInfo ni;
            if (parse_node_proto(raw.data(), raw.size(), ni) && !ni.op_type.empty())
                g.nodes.push_back(std::move(ni));
        } else if (fn == 5) {  // initializer (repeated TensorProto)
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            TensorInit ti;
            if (parse_tensor_proto(raw.data(), raw.size(), ti) && !ti.name.empty())
                g.initializers[ti.name] = std::move(ti);
        } else if (fn == 11) {  // input (ValueInfoProto) - first one is usually the graph input
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            if (g.input_shape.empty())
                parse_value_info_shape(raw.data(), raw.size(), g.input_shape);
        } else {
            r.skip_field(wt);
        }
    }
    return true;
}

// Parse ModelProto: extract graph (field 7)
bool parse_model_proto(const uint8_t* buf, size_t len, GraphState& g) {
    ProtobufReader r;
    r.reset(buf, len);
    while (!r.eof()) {
        int fn, wt;
        if (r.read_tag(fn, wt) != 0) break;
        if (fn == 7) {  // graph
            std::vector<uint8_t> raw;
            if (!r.read_length_delimited(raw)) return false;
            return parse_graph_proto(raw.data(), raw.size(), g);
        }
        r.skip_field(wt);
    }
    return false;
}

// =============================================================================
// Build Sequential from GraphState (same op set as export)
// =============================================================================

bool build_sequential(const GraphState& g, Sequential* model) {
    std::vector<int64_t> current_shape = g.input_shape;
    if (current_shape.empty()) return false;

    for (const NodeInfo& node : g.nodes) {
        if (node.op_type == "Gemm") {
            // inputs: [input, weight, bias]
            auto it_w = g.initializers.find(node.inputs.size() > 1 ? node.inputs[1] : "");
            auto it_b = g.initializers.find(node.inputs.size() > 2 ? node.inputs[2] : "");
            if (it_w == g.initializers.end() || it_b == g.initializers.end()) return false;
            const TensorInit& w = it_w->second;
            const TensorInit& b = it_b->second;
            if (w.dims.size() != 2 || w.float_data.size() != static_cast<size_t>(w.dims[0] * w.dims[1])) return false;
            size_t in_f = static_cast<size_t>(w.dims[0]);
            size_t out_f = static_cast<size_t>(w.dims[1]);
            Linear* lin = new Linear(in_f, out_f);
            memcpy(lin->weight->data(), w.float_data.data(), w.float_data.size() * sizeof(float));
            memcpy(lin->bias->data(), b.float_data.data(), b.float_data.size() * sizeof(float));
            model->add(lin);
            current_shape.back() = static_cast<int64_t>(out_f);
        } else if (node.op_type == "Conv") {
            auto it_w = g.initializers.find(node.inputs.size() > 1 ? node.inputs[1] : "");
            auto it_b = g.initializers.find(node.inputs.size() > 2 ? node.inputs[2] : "");
            if (it_w == g.initializers.end() || it_b == g.initializers.end()) return false;
            const TensorInit& w = it_w->second;
            const TensorInit& b = it_b->second;
            if (w.dims.size() != 4) return false;
            int64_t out_c = w.dims[0], in_c = w.dims[1], k = w.dims[2];
            if (w.dims[3] != k) return false;
            std::vector<int64_t> pads = {0, 0, 0, 0};
            int64_t stride = 1;
            auto pit = node.attrs_ints.find("pads");
            if (pit != node.attrs_ints.end() && pit->second.size() >= 4)
                pads = pit->second;
            auto sit = node.attrs_ints.find("strides");
            if (sit != node.attrs_ints.end() && !sit->second.empty())
                stride = sit->second[0];
            Conv2d* conv = new Conv2d(static_cast<size_t>(in_c), static_cast<size_t>(out_c),
                                     static_cast<size_t>(k), static_cast<size_t>(stride),
                                     static_cast<size_t>(pads[0]));
            memcpy(conv->weight->data(), w.float_data.data(), w.float_data.size() * sizeof(float));
            memcpy(conv->bias->data(), b.float_data.data(), b.float_data.size() * sizeof(float));
            model->add(conv);
            int64_t h = current_shape[2], ww = current_shape[3];
            current_shape = {current_shape[0], out_c,
                            (h + 2 * pads[0] - k) / stride + 1,
                            (ww + 2 * pads[1] - k) / stride + 1};
        } else if (node.op_type == "Relu") {
            model->add(new ReLU());
        } else if (node.op_type == "Sigmoid") {
            model->add(new Sigmoid());
        } else if (node.op_type == "Tanh") {
            model->add(new Tanh());
        } else if (node.op_type == "Softmax") {
            int axis = 1;
            auto it = node.attrs_i.find("axis");
            if (it != node.attrs_i.end()) axis = static_cast<int>(it->second);
            model->add(new Softmax(axis));
        } else if (node.op_type == "Flatten") {
            model->add(new Flatten());
            int64_t flat = 1;
            for (size_t i = 1; i < current_shape.size(); i++) flat *= current_shape[i];
            current_shape = {current_shape[0], flat};
        } else if (node.op_type == "MaxPool") {
            int64_t k = 2, s = 2;
            auto kit = node.attrs_ints.find("kernel_shape");
            auto sit = node.attrs_ints.find("strides");
            if (kit != node.attrs_ints.end() && !kit->second.empty()) k = kit->second[0];
            if (sit != node.attrs_ints.end() && !sit->second.empty()) s = sit->second[0];
            model->add(new MaxPool2d(static_cast<size_t>(k), static_cast<size_t>(s)));
            current_shape[2] /= s;
            current_shape[3] /= s;
        } else if (node.op_type == "AveragePool") {
            int64_t k = 2, s = 2;
            auto kit = node.attrs_ints.find("kernel_shape");
            auto sit = node.attrs_ints.find("strides");
            if (kit != node.attrs_ints.end() && !kit->second.empty()) k = kit->second[0];
            if (sit != node.attrs_ints.end() && !sit->second.empty()) s = sit->second[0];
            model->add(new AvgPool2d(static_cast<size_t>(k), static_cast<size_t>(s)));
            current_shape[2] /= s;
            current_shape[3] /= s;
        } else if (node.op_type == "BatchNormalization") {
            if (node.inputs.size() < 5) return false;
            std::string scale_name = node.inputs[1], bias_name = node.inputs[2],
                        mean_name = node.inputs[3], var_name = node.inputs[4];
            auto it_s = g.initializers.find(scale_name);
            auto it_b = g.initializers.find(bias_name);
            auto it_m = g.initializers.find(mean_name);
            auto it_v = g.initializers.find(var_name);
            if (it_s == g.initializers.end() || it_b == g.initializers.end() ||
                it_m == g.initializers.end() || it_v == g.initializers.end()) return false;
            size_t num_f = it_s->second.float_data.size();
            float eps = 1e-5f;
            auto eit = node.attrs_f.find("epsilon");
            if (eit != node.attrs_f.end()) eps = eit->second;
            BatchNorm2d* bn = new BatchNorm2d(num_f, eps);
            memcpy(bn->gamma->data(), it_s->second.float_data.data(), num_f * sizeof(float));
            memcpy(bn->beta->data(), it_b->second.float_data.data(), num_f * sizeof(float));
            memcpy(bn->running_mean->data(), it_m->second.float_data.data(), num_f * sizeof(float));
            memcpy(bn->running_var->data(), it_v->second.float_data.data(), num_f * sizeof(float));
            bn->eval();
            model->add(bn);
        } else if (node.op_type == "Identity") {
            // Dropout in inference -> Identity; we can skip or add no-op. Skip for simplicity.
        } else {
            fprintf(stderr, "ONNX import: unsupported op_type '%s'\n", node.op_type.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

// =============================================================================
// Public API
// =============================================================================

std::unique_ptr<Sequential> load_onnx(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "ONNX import: cannot open '%s'\n", filepath.c_str());
        return nullptr;
    }
    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint8_t> buf(file_size);
    if (!file.read(reinterpret_cast<char*>(buf.data()), file_size)) {
        fprintf(stderr, "ONNX import: read failed '%s'\n", filepath.c_str());
        return nullptr;
    }
    file.close();

    GraphState g;
    if (!parse_model_proto(buf.data(), buf.size(), g)) {
        fprintf(stderr, "ONNX import: parse failed '%s'\n", filepath.c_str());
        return nullptr;
    }
    if (g.nodes.empty()) {
        fprintf(stderr, "ONNX import: no nodes in graph '%s'\n", filepath.c_str());
        return nullptr;
    }

    auto model = std::make_unique<Sequential>();
    if (!build_sequential(g, model.get())) {
        fprintf(stderr, "ONNX import: build failed '%s'\n", filepath.c_str());
        return nullptr;
    }
    model->eval();
    return model;
}

bool load_onnx_input_shape(const std::string& filepath, std::vector<size_t>& out_shape) {
    out_shape.clear();
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return false;
    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint8_t> buf(file_size);
    if (!file.read(reinterpret_cast<char*>(buf.data()), file_size)) return false;
    file.close();

    GraphState g;
    if (!parse_model_proto(buf.data(), buf.size(), g) || g.input_shape.empty()) return false;
    for (int64_t d : g.input_shape)
        out_shape.push_back(static_cast<size_t>(d));
    return true;
}
