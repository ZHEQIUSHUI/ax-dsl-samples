#pragma once

#include <cstddef>
#include <vector>

#include <ax_engine_api.h>

namespace ax_dsl {

inline constexpr float kMaxError = 1e-4f;

void print_values(const char* name, const float* values, std::size_t count);

// Small wrapper around the AX Engine lifecycle used by every sample.
class AxEngineRunner {
public:
    AxEngineRunner() = default;
    ~AxEngineRunner();

    AxEngineRunner(const AxEngineRunner&) = delete;
    AxEngineRunner& operator=(const AxEngineRunner&) = delete;

    bool init();
    bool load_model(const char* model_path);
    bool allocate_io();
    bool run();
    void release();

    float* input(std::size_t index);
    const float* output(std::size_t index) const;
    AX_U32 input_count() const;
    std::size_t element_count() const;

private:
    bool allocate_buffer(AX_ENGINE_IO_BUFFER_T* buffer, const AX_ENGINE_IOMETA_T& meta);

    std::vector<char> model_buffer_;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* io_info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    std::size_t element_count_ = 0;
    bool sys_initialized_ = false;
    bool engine_initialized_ = false;
};

} // namespace ax_dsl
