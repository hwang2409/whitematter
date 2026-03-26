#include "../layer.h"
#include <cassert>
#include <cmath>
#include <string>
#include <algorithm>

Upsample::Upsample(size_t scale_factor, std::string mode)
    : scale_factor(scale_factor), mode(std::move(mode)) {
    assert((this->mode == "nearest" || this->mode == "bilinear") &&
           "Upsample mode must be 'nearest' or 'bilinear'");
    assert(scale_factor >= 1);
}

TensorPtr Upsample::forward(const TensorPtr& input) {
    return input->upsample(scale_factor, mode);
}

std::string Upsample::extra_repr() const {
    return "scale_factor=" + std::to_string(scale_factor) + ", mode=" + mode;
}

std::vector<size_t> Upsample::compute_output_shape(const std::vector<size_t>& input_shape) const {
    if (input_shape.size() != 4) return input_shape;
    return {input_shape[0], input_shape[1],
            input_shape[2] * scale_factor, input_shape[3] * scale_factor};
}
