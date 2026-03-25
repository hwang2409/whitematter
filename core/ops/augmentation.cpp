#include "../tensor.h"
#include <algorithm>
#include <random>
#include <cassert>

TensorPtr Tensor::flip_horizontal() const {
    assert(shape.size() == 3 || shape.size() == 4);

    size_t batch = 1, channels, height, width;
    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
    }

    auto result = Tensor::create(shape, false);

    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width + h * width + w;
                        dst_idx = n * channels * height * width + c * height * width + h * width + (width - 1 - w);
                    } else {
                        src_idx = c * height * width + h * width + w;
                        dst_idx = c * height * width + h * width + (width - 1 - w);
                    }
                    result->data()[dst_idx] = data()[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr Tensor::random_flip_horizontal(float p) const {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (shape.size() == 4) {
        size_t batch = shape[0];
        size_t channels = shape[1];
        size_t height = shape[2];
        size_t width = shape[3];
        size_t img_size = channels * height * width;

        auto result = Tensor::create(shape, false);

        for (size_t n = 0; n < batch; n++) {
            bool do_flip = dist(gen) < p;
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        size_t src_idx = n * img_size + c * height * width + h * width + w;
                        size_t dst_w = do_flip ? (width - 1 - w) : w;
                        size_t dst_idx = n * img_size + c * height * width + h * width + dst_w;
                        result->data()[dst_idx] = data()[src_idx];
                    }
                }
            }
        }
        return result;
    } else {
        if (dist(gen) < p) {
            return flip_horizontal();
        } else {
            auto result = Tensor::create(shape, false);
            std::copy(data(), data() + size(), result->data());
            return result;
        }
    }
}

TensorPtr Tensor::pad2d(size_t padding) const {
    assert(shape.size() == 3 || shape.size() == 4);

    size_t batch = 1, channels, height, width;
    std::vector<size_t> new_shape;

    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
        new_shape = {batch, channels, height + 2 * padding, width + 2 * padding};
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
        new_shape = {channels, height + 2 * padding, width + 2 * padding};
    }

    size_t new_height = height + 2 * padding;
    size_t new_width = width + 2 * padding;

    auto result = Tensor::zeros(new_shape, false);

    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width + h * width + w;
                        dst_idx = n * channels * new_height * new_width + c * new_height * new_width +
                                  (h + padding) * new_width + (w + padding);
                    } else {
                        src_idx = c * height * width + h * width + w;
                        dst_idx = c * new_height * new_width + (h + padding) * new_width + (w + padding);
                    }
                    result->data()[dst_idx] = data()[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr Tensor::crop(size_t top, size_t left, size_t crop_height, size_t crop_width) const {
    assert(shape.size() == 3 || shape.size() == 4);

    size_t batch = 1, channels, height, width;
    std::vector<size_t> new_shape;

    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
        new_shape = {batch, channels, crop_height, crop_width};
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
        new_shape = {channels, crop_height, crop_width};
    }

    assert(top + crop_height <= height && "Crop exceeds image height");
    assert(left + crop_width <= width && "Crop exceeds image width");

    auto result = Tensor::create(new_shape, false);

    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_height; h++) {
                for (size_t w = 0; w < crop_width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width +
                                  (top + h) * width + (left + w);
                        dst_idx = n * channels * crop_height * crop_width + c * crop_height * crop_width +
                                  h * crop_width + w;
                    } else {
                        src_idx = c * height * width + (top + h) * width + (left + w);
                        dst_idx = c * crop_height * crop_width + h * crop_width + w;
                    }
                    result->data()[dst_idx] = data()[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr Tensor::random_crop(size_t crop_height, size_t crop_width) const {
    assert(shape.size() == 3 || shape.size() == 4);

    static thread_local std::mt19937 gen(std::random_device{}());

    size_t batch = 1, channels, height, width;
    std::vector<size_t> new_shape;

    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
        new_shape = {batch, channels, crop_height, crop_width};
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
        new_shape = {channels, crop_height, crop_width};
    }

    assert(crop_height <= height && "Crop height exceeds image height");
    assert(crop_width <= width && "Crop width exceeds image width");

    auto result = Tensor::create(new_shape, false);

    std::uniform_int_distribution<size_t> top_dist(0, height - crop_height);
    std::uniform_int_distribution<size_t> left_dist(0, width - crop_width);

    for (size_t n = 0; n < batch; n++) {
        size_t top = top_dist(gen);
        size_t left = left_dist(gen);

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_height; h++) {
                for (size_t w = 0; w < crop_width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width +
                                  (top + h) * width + (left + w);
                        dst_idx = n * channels * crop_height * crop_width + c * crop_height * crop_width +
                                  h * crop_width + w;
                    } else {
                        src_idx = c * height * width + (top + h) * width + (left + w);
                        dst_idx = c * crop_height * crop_width + h * crop_width + w;
                    }
                    result->data()[dst_idx] = data()[src_idx];
                }
            }
        }
    }

    return result;
}
