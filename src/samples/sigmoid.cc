#include <algorithm>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

namespace {

constexpr AX_U32 kInputCount = 1;

bool verify(const ax_dsl::SampleContext& context, float& max_error) {
    if (!ax_dsl::has_shape(context.output_meta, {256, 256})) {
        std::fprintf(stderr, "sigmoid expects output shape 256x256\n");
        return false;
    }

    const float* x = context.inputs[0];
    for (std::size_t i = 0; i < context.element_count; ++i) {
        const double expected = 1.0 / (1.0 + std::exp(-static_cast<double>(x[i])));
        max_error = std::max(max_error, std::fabs(context.output[i] - static_cast<float>(expected)));
    }
    return true;
}

const ax_dsl::SampleDefinition kSample{"sigmoid", kInputCount, verify};

} // namespace

int main(int argc, char** argv) { return ax_dsl::run_sample_cli(argc, argv, kSample); }
