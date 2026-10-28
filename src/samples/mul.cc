#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

// Demo: multiply two eight-element float32 vectors and verify the NPU result.
namespace {

constexpr AX_U32 kInputCount = 2;
constexpr AX_U32 kElementCount = 8;
constexpr std::array<float, kElementCount> kA = {1.0f, 2.0f, 3.0f, 4.0f,
                                                  5.0f, 6.0f, 7.0f, 8.0f};
constexpr std::array<float, kElementCount> kB = {10.0f, 20.0f, 30.0f, 40.0f,
                                                  50.0f, 60.0f, 70.0f, 80.0f};

void fill(float* const* inputs, std::size_t element_count) {
    std::copy_n(kA.data(), element_count, inputs[0]);
    std::copy_n(kB.data(), element_count, inputs[1]);
}

bool verify(const ax_dsl::SampleContext& context, float& max_error) {
    if (!ax_dsl::has_shape(context.output_meta, {kElementCount})) {
        std::fprintf(stderr, "mul expects an 8-element vector\n");
        return false;
    }

    const float* a = context.inputs[0];
    const float* b = context.inputs[1];
    for (std::size_t i = 0; i < context.element_count; ++i) {
        max_error = std::max(max_error, std::fabs(context.output[i] - (a[i] * b[i])));
    }
    return true;
}

const ax_dsl::SampleDefinition kSample{"mul", kInputCount, kElementCount, fill, verify};

} // namespace

int main(int argc, char** argv) { return ax_dsl::run_sample_cli(argc, argv, kSample); }
