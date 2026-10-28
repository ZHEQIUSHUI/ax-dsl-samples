#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

// Demo: apply GELU to an eight-element float32 vector and verify the NPU result.
namespace {

constexpr AX_U32 kInputCount = 1;
constexpr AX_U32 kElementCount = 8;
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr std::array<float, kElementCount> kX = {-3.0f, -2.0f, -1.0f, 0.0f,
                                                  1.0f,  2.0f,  3.0f,  4.0f};

void fill(float* const* inputs, std::size_t element_count) {
    std::copy_n(kX.data(), element_count, inputs[0]);
}

bool verify(const ax_dsl::SampleContext& context, float& max_error) {
    if (!ax_dsl::has_shape(context.output_meta, {kElementCount})) {
        std::fprintf(stderr, "gelu expects an 8-element vector\n");
        return false;
    }

    const float* x = context.inputs[0];
    for (std::size_t i = 0; i < context.element_count; ++i) {
        const double value = static_cast<double>(x[i]);
        const double expected = 0.5 * value * (1.0 + std::erf(value * kInvSqrt2));
        max_error = std::max(max_error, std::fabs(context.output[i] - static_cast<float>(expected)));
    }
    return true;
}

const ax_dsl::SampleDefinition kSample{"gelu", kInputCount, kElementCount, fill, verify};

} // namespace

int main(int argc, char** argv) { return ax_dsl::run_sample_cli(argc, argv, kSample); }
