#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

namespace {

constexpr std::size_t kInputCount = 1;
constexpr std::size_t kElementCount = 8;
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr std::array<float, kElementCount> kX = {-3.0f, -2.0f, -1.0f, 0.0f,
                                                  1.0f,  2.0f,  3.0f,  4.0f};

float reference_error(const float* x, const float* output) {
    float max_error = 0.0f;
    for (std::size_t i = 0; i < kElementCount; ++i) {
        const double value = static_cast<double>(x[i]);
        const double expected = 0.5 * value * (1.0 + std::erf(value * kInvSqrt2));
        max_error = std::max(max_error, std::fabs(output[i] - static_cast<float>(expected)));
    }
    return max_error;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <model.axmodel>\n", argv[0]);
        return 1;
    }

    ax_dsl::AxEngineRunner runner;

    // 1. Initialize AX system services and the NPU engine.
    if (!runner.init()) {
        return 1;
    }

    // 2. Load the model and allocate CMM buffers for its input and output.
    if (!runner.load_model(argv[1])) {
        return 1;
    }
    if (!runner.allocate_io()) {
        return 1;
    }
    if (runner.input_count() != kInputCount || runner.element_count() != kElementCount) {
        std::fprintf(stderr, "model IO does not match this gelu sample\n");
        return 1;
    }

    // 3. Copy the demo vector into the model input buffer.
    float* x = runner.input(0);
    std::copy(kX.begin(), kX.end(), x);
    ax_dsl::print_values("x", x, kElementCount);

    // 4. Run the model on the NPU.
    if (!runner.run()) {
        return 1;
    }

    // 5. Read the output buffer and print the GELU result.
    const float* output = runner.output(0);
    ax_dsl::print_values("gelu(x)", output, kElementCount);

    // 6. Compare the NPU result with a CPU calculation.
    const float max_error = reference_error(x, output);
    if (max_error > ax_dsl::kMaxError) {
        std::fprintf(stderr, "gelu verification failed, max error: %g\n", max_error);
        return 1;
    }
    std::printf("AX Engine gelu sample: PASS (%zu elements, max error %.6g)\n",
                kElementCount, max_error);

    runner.release();
    return 0;
}
