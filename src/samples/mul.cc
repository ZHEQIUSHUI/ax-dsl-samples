#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "ax_dsl_runner.h"

namespace {

constexpr std::size_t kInputCount = 2;
constexpr std::size_t kElementCount = 8;
constexpr std::array<float, kElementCount> kA = {1.0f, 2.0f, 3.0f, 4.0f,
                                                  5.0f, 6.0f, 7.0f, 8.0f};
constexpr std::array<float, kElementCount> kB = {10.0f, 20.0f, 30.0f, 40.0f,
                                                  50.0f, 60.0f, 70.0f, 80.0f};

float reference_error(const float* a, const float* b, const float* output) {
    float max_error = 0.0f;
    for (std::size_t i = 0; i < kElementCount; ++i) {
        max_error = std::max(max_error, std::fabs(output[i] - (a[i] * b[i])));
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

    // 2. Load the model and allocate CMM buffers for its inputs and output.
    if (!runner.load_model(argv[1])) {
        return 1;
    }
    if (!runner.allocate_io()) {
        return 1;
    }
    if (runner.input_count() != kInputCount || runner.element_count() != kElementCount) {
        std::fprintf(stderr, "model IO does not match this mul sample\n");
        return 1;
    }

    // 3. Copy the two demo vectors into the model input buffers.
    float* a = runner.input(0);
    float* b = runner.input(1);
    std::copy(kA.begin(), kA.end(), a);
    std::copy(kB.begin(), kB.end(), b);
    ax_dsl::print_values("a", a, kElementCount);
    ax_dsl::print_values("b", b, kElementCount);

    // 4. Run the model on the NPU.
    if (!runner.run()) {
        return 1;
    }

    // 5. Read the output buffer and print the multiplication result.
    const float* output = runner.output(0);
    ax_dsl::print_values("a * b", output, kElementCount);

    // 6. Compare the NPU result with a CPU calculation.
    const float max_error = reference_error(a, b, output);
    if (max_error > ax_dsl::kMaxError) {
        std::fprintf(stderr, "mul verification failed, max error: %g\n", max_error);
        return 1;
    }
    std::printf("AX Engine mul sample: PASS (%zu elements, max error %.6g)\n",
                kElementCount, max_error);

    runner.release();
    return 0;
}
