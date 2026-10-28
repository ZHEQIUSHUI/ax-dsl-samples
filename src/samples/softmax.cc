#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

// Demo: convert eight scores into probabilities and verify the NPU result.
namespace {

constexpr AX_U32 kInputCount = 1;
constexpr AX_U32 kElementCount = 8;
constexpr std::array<float, kElementCount> kX = {-3.0f, -2.0f, -1.0f, 0.0f,
                                                  1.0f,  2.0f,  3.0f,  4.0f};

void fill(float* const* inputs, std::size_t element_count) {
    std::copy_n(kX.data(), element_count, inputs[0]);
}

bool verify(const ax_dsl::SampleContext& context, float& max_error) {
    if (!ax_dsl::has_shape(context.output_meta, {kElementCount})) {
        std::fprintf(stderr, "softmax expects an 8-element vector\n");
        return false;
    }

    const AX_U32 last_dim = context.output_meta.pShape[context.output_meta.nShapeSize - 1];
    const std::size_t rows = context.element_count / last_dim;
    const float* x = context.inputs[0];
    for (std::size_t row = 0; row < rows; ++row) {
        const float* values = x + row * last_dim;
        const float row_max = *std::max_element(values, values + last_dim);
        double sum = 0.0;
        for (AX_U32 col = 0; col < last_dim; ++col) {
            sum += std::exp(static_cast<double>(values[col] - row_max));
        }
        for (AX_U32 col = 0; col < last_dim; ++col) {
            const double expected = std::exp(static_cast<double>(values[col] - row_max)) / sum;
            max_error =
                std::max(max_error, std::fabs(context.output[row * last_dim + col] -
                                              static_cast<float>(expected)));
        }
    }
    return true;
}

const ax_dsl::SampleDefinition kSample{"softmax", kInputCount, kElementCount, fill, verify};

} // namespace

int main(int argc, char** argv) { return ax_dsl::run_sample_cli(argc, argv, kSample); }
