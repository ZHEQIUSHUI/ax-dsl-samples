#include <algorithm>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

namespace {

constexpr AX_U32 kInputCount = 1;
constexpr double kInvSqrt2 = 0.70710678118654752440;

bool verify(const ax_dsl::SampleContext& context, float& max_error) {
    if (!ax_dsl::has_shape(context.output_meta, {256, 256})) {
        std::fprintf(stderr, "gelu expects output shape 256x256\n");
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

const ax_dsl::SampleDefinition kSample{"gelu", kInputCount, verify};

} // namespace

int main(int argc, char** argv) { return ax_dsl::run_sample_cli(argc, argv, kSample); }
