#pragma once

#include <cstddef>
#include <initializer_list>

#include <ax_engine_api.h>

namespace ax_dsl {

struct SampleContext {
    const float* const* inputs;
    const float* output;
    std::size_t element_count;
    const AX_ENGINE_IOMETA_T& output_meta;
};

using VerifyFn = bool (*)(const SampleContext& context, float& max_error);
using FillFn = void (*)(float* const* inputs, std::size_t element_count);

struct SampleDefinition {
    const char* name;
    AX_U32 input_count;
    std::size_t element_count;
    FillFn fill;
    VerifyFn verify;
};

bool has_shape(const AX_ENGINE_IOMETA_T& meta,
               std::initializer_list<AX_U32> expected_shape);

int run_sample_cli(int argc, char** argv, const SampleDefinition& sample);

} // namespace ax_dsl
