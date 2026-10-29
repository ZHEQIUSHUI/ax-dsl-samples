#include "ax_dsl_runner.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include <ax_sys_api.h>

namespace ax_dsl {
namespace {

constexpr AX_U32 kCmmAlignSize = 128;
constexpr const char* kCmmToken = "ax_dsl_runner";

bool read_file(const char* path, std::vector<char>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size));
    return static_cast<bool>(file.read(data.data(), size));
}

const char* dtype_name(AX_ENGINE_DATA_TYPE_T dtype) {
    switch (dtype) {
    case AX_ENGINE_DT_FLOAT32:
        return "FLOAT32";
    case AX_ENGINE_DT_FLOAT16:
        return "FLOAT16";
    case AX_ENGINE_DT_BFLOAT16:
        return "BFLOAT16";
    case AX_ENGINE_DT_UINT8:
        return "UINT8";
    case AX_ENGINE_DT_SINT8:
        return "SINT8";
    case AX_ENGINE_DT_SINT32:
        return "SINT32";
    default:
        return "OTHER";
    }
}

void print_shape(const AX_ENGINE_IOMETA_T& meta) {
    for (AX_U8 i = 0; i < meta.nShapeSize; ++i) {
        std::printf("%s%d", i == 0 ? "" : "x", meta.pShape[i]);
    }
}

void print_io_info(const AX_ENGINE_IO_INFO_T& info) {
    std::printf("inputs: %u\n", info.nInputSize);
    for (AX_U32 i = 0; i < info.nInputSize; ++i) {
        const AX_ENGINE_IOMETA_T& meta = info.pInputs[i];
        std::printf("  [%u] %s %s ", i, meta.pName != nullptr ? meta.pName : "<unnamed>",
                    dtype_name(meta.eDataType));
        print_shape(meta);
        std::printf(" (%u bytes)\n", meta.nSize);
    }

    std::printf("outputs: %u\n", info.nOutputSize);
    for (AX_U32 i = 0; i < info.nOutputSize; ++i) {
        const AX_ENGINE_IOMETA_T& meta = info.pOutputs[i];
        std::printf("  [%u] %s %s ", i, meta.pName != nullptr ? meta.pName : "<unnamed>",
                    dtype_name(meta.eDataType));
        print_shape(meta);
        std::printf(" (%u bytes)\n", meta.nSize);
    }
}

void print_error(const char* api, AX_S32 ret) {
    std::fprintf(stderr, "%s failed: 0x%08x\n", api, static_cast<unsigned>(ret));
}

} // namespace

void print_values(const char* name, const float* values, std::size_t count) {
    std::printf("%s: [", name != nullptr ? name : "<unnamed>");
    for (std::size_t i = 0; i < count; ++i) {
        std::printf("%s%.6g", i == 0 ? "" : ", ", values[i]);
    }
    std::printf("]\n");
}

AxEngineRunner::~AxEngineRunner() { release(); }

bool AxEngineRunner::init() {
    // Initialize AX system services: CMM allocator, device access, and logging.
    AX_S32 ret = AX_SYS_Init();
    if (ret != 0) {
        print_error("AX_SYS_Init", ret);
        return false;
    }
    sys_initialized_ = true;

    // Initialize the NPU inference engine.
    AX_ENGINE_NPU_ATTR_T npu_attr;
    std::memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        print_error("AX_ENGINE_Init", ret);
        return false;
    }
    engine_initialized_ = true;

    std::printf("AX Engine version: %s\n", AX_ENGINE_GetVersion());
    return true;
}

bool AxEngineRunner::load_model(const char* model_path) {
    // Keep the model bytes alive for as long as the inference handle exists.
    if (!read_file(model_path, model_buffer_)) {
        std::fprintf(stderr, "failed to read AXModel: %s\n", model_path);
        return false;
    }
    if (model_buffer_.size() > std::numeric_limits<AX_U32>::max()) {
        std::fprintf(stderr, "AXModel is too large: %zu bytes\n", model_buffer_.size());
        return false;
    }

    AX_S32 ret = AX_ENGINE_CreateHandle(&handle_, model_buffer_.data(),
                                        static_cast<AX_U32>(model_buffer_.size()));
    if (ret != 0) {
        print_error("AX_ENGINE_CreateHandle", ret);
        return false;
    }

    ret = AX_ENGINE_CreateContext(handle_);
    if (ret != 0) {
        print_error("AX_ENGINE_CreateContext", ret);
        return false;
    }

    ret = AX_ENGINE_GetIOInfo(handle_, &io_info_);
    if (ret != 0) {
        print_error("AX_ENGINE_GetIOInfo", ret);
        return false;
    }
    print_io_info(*io_info_);

    if (io_info_->nOutputSize != 1) {
        std::fprintf(stderr, "this demo expects exactly one output tensor\n");
        return false;
    }

    const AX_ENGINE_IOMETA_T& output_meta = io_info_->pOutputs[0];
    if (output_meta.eDataType != AX_ENGINE_DT_FLOAT32 ||
        output_meta.nSize % sizeof(float) != 0) {
        std::fprintf(stderr, "this demo expects a float32 output tensor\n");
        return false;
    }
    element_count_ = output_meta.nSize / sizeof(float);

    for (AX_U32 i = 0; i < io_info_->nInputSize; ++i) {
        const AX_ENGINE_IOMETA_T& input_meta = io_info_->pInputs[i];
        if (input_meta.eDataType != AX_ENGINE_DT_FLOAT32 ||
            input_meta.nSize != output_meta.nSize) {
            std::fprintf(stderr,
                         "this demo expects float32 input/output tensors with equal sizes\n");
            return false;
        }
    }
    return true;
}

bool AxEngineRunner::allocate_io() {
    // Allocate one physically contiguous CMM buffer for every model input.
    io_.nInputSize = io_info_->nInputSize;
    io_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_.nInputSize]{};
    for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
        if (!allocate_buffer(&io_.pInputs[i], io_info_->pInputs[i])) {
            return false;
        }
    }

    // Allocate one CMM buffer for every model output.
    io_.nOutputSize = io_info_->nOutputSize;
    io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_.nOutputSize]{};
    for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
        if (!allocate_buffer(&io_.pOutputs[i], io_info_->pOutputs[i])) {
            return false;
        }
    }
    return true;
}

bool AxEngineRunner::run() {
    AX_S32 ret = AX_ENGINE_RunSync(handle_, &io_);
    if (ret != 0) {
        print_error("AX_ENGINE_RunSync", ret);
        return false;
    }
    return true;
}

float* AxEngineRunner::input(std::size_t index) {
    return static_cast<float*>(io_.pInputs[index].pVirAddr);
}

const float* AxEngineRunner::output(std::size_t index) const {
    return static_cast<const float*>(io_.pOutputs[index].pVirAddr);
}

AX_U32 AxEngineRunner::input_count() const {
    return io_info_ != nullptr ? io_info_->nInputSize : 0;
}

std::size_t AxEngineRunner::element_count() const { return element_count_; }

bool AxEngineRunner::allocate_buffer(AX_ENGINE_IO_BUFFER_T* buffer,
                                     const AX_ENGINE_IOMETA_T& meta) {
    std::memset(buffer, 0, sizeof(*buffer));
    buffer->nSize = meta.nSize;
    AX_S32 ret = AX_SYS_MemAlloc(&buffer->phyAddr, &buffer->pVirAddr, meta.nSize, kCmmAlignSize,
                                 reinterpret_cast<const AX_S8*>(kCmmToken));
    if (ret != 0) {
        print_error("AX_SYS_MemAlloc", ret);
        return false;
    }
    return true;
}

void AxEngineRunner::release() {
    // Free CMM buffers before destroying the engine resources that use them.
    if (io_.pInputs != nullptr) {
        for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
            if (io_.pInputs[i].pVirAddr != nullptr) {
                AX_SYS_MemFree(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr);
            }
        }
        delete[] io_.pInputs;
        io_.pInputs = nullptr;
    }
    if (io_.pOutputs != nullptr) {
        for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
            if (io_.pOutputs[i].pVirAddr != nullptr) {
                AX_SYS_MemFree(io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr);
            }
        }
        delete[] io_.pOutputs;
        io_.pOutputs = nullptr;
    }

    // Destroy the model handle, then finalize the NPU engine and system layer.
    if (handle_ != nullptr) {
        AX_ENGINE_DestroyHandle(handle_);
        handle_ = nullptr;
    }
    if (engine_initialized_) {
        AX_ENGINE_Deinit();
        engine_initialized_ = false;
    }
    if (sys_initialized_) {
        AX_SYS_Deinit();
        sys_initialized_ = false;
    }
}

} // namespace ax_dsl
