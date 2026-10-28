#include <algorithm>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

namespace {

constexpr AX_U32 kInputCount = 2;

bool verify(const ax_dsl::SampleContext& context, float& max_error) {
    if (!ax_dsl::has_shape(context.output_meta, {256, 512})) {
        std::fprintf(stderr, "mul expects output shape 256x512\n");
        return false;
    }

    const float* a = context.inputs[0];
    const float* b = context.inputs[1];
    for (std::size_t i = 0; i < context.element_count; ++i) {
        max_error = std::max(max_error, std::fabs(context.output[i] - (a[i] * b[i])));
    }
    return true;
}

const ax_dsl::SampleDefinition kSample{"mul", kInputCount, verify};

} // namespace

int main(int argc, char** argv) { return ax_dsl::run_sample_cli(argc, argv, kSample); }
