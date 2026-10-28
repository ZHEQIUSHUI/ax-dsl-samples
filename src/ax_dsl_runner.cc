#include "ax_dsl_runner.h"

#include <algorithm>
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
constexpr float kMaxError = 1e-4f;

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

void print_values(const char* name, const float* values, std::size_t count) {
    std::printf("%s: [", name != nullptr ? name : "<unnamed>");
    for (std::size_t i = 0; i < count; ++i) {
        std::printf("%s%.6g", i == 0 ? "" : ", ", values[i]);
    }
    std::printf("]\n");
}

class AxEngineRunner {
public:
    explicit AxEngineRunner(const SampleDefinition& sample) : sample_(sample) {}
    ~AxEngineRunner() { release(); }

    AxEngineRunner(const AxEngineRunner&) = delete;
    AxEngineRunner& operator=(const AxEngineRunner&) = delete;

    bool run(const char* model_path) {
        return init_system() && load_model(model_path) && allocate_io() && fill_inputs() &&
               infer() && report() && verify();
    }

private:
    bool init_system() {
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
        std::printf("sample: %s\n", sample_.name);
        return true;
    }

    bool load_model(const char* model_path) {
        // Read the AXModel file into memory. The buffer stays alive while the handle exists.
        if (!read_file(model_path, model_buffer_)) {
            std::fprintf(stderr, "failed to read AXModel: %s\n", model_path);
            return false;
        }
        if (model_buffer_.size() > std::numeric_limits<AX_U32>::max()) {
            std::fprintf(stderr, "AXModel is too large: %zu bytes\n", model_buffer_.size());
            return false;
        }

        // Create an inference handle from the AXModel.
        AX_S32 ret = AX_ENGINE_CreateHandle(&handle_, model_buffer_.data(),
                                            static_cast<AX_U32>(model_buffer_.size()));
        if (ret != 0) {
            print_error("AX_ENGINE_CreateHandle", ret);
            return false;
        }

        // Create the default execution context for this handle.
        ret = AX_ENGINE_CreateContext(handle_);
        if (ret != 0) {
            print_error("AX_ENGINE_CreateContext", ret);
            return false;
        }

        // Query tensor names, shapes, dtypes, and required buffer sizes.
        ret = AX_ENGINE_GetIOInfo(handle_, &io_info_);
        if (ret != 0) {
            print_error("AX_ENGINE_GetIOInfo", ret);
            return false;
        }
        print_io_info(*io_info_);

        // Validate the common float32 IO contract shared by all samples.
        if (io_info_->nInputSize != sample_.input_count || io_info_->nOutputSize != 1) {
            std::fprintf(stderr, "sample %s expects %u input(s) and 1 output\n", sample_.name,
                         sample_.input_count);
            return false;
        }

        const AX_ENGINE_IOMETA_T& output_meta = io_info_->pOutputs[0];
        if (output_meta.eDataType != AX_ENGINE_DT_FLOAT32 ||
            output_meta.nSize % sizeof(float) != 0) {
            std::fprintf(stderr, "sample %s expects a float32 output tensor\n", sample_.name);
            return false;
        }
        element_count_ = output_meta.nSize / sizeof(float);
        if (element_count_ != sample_.element_count) {
            std::fprintf(stderr, "sample %s expects %zu elements, model has %zu\n", sample_.name,
                         sample_.element_count, element_count_);
            return false;
        }

        for (AX_U32 i = 0; i < io_info_->nInputSize; ++i) {
            const AX_ENGINE_IOMETA_T& input_meta = io_info_->pInputs[i];
            if (input_meta.eDataType != AX_ENGINE_DT_FLOAT32 ||
                input_meta.nSize != output_meta.nSize) {
                std::fprintf(stderr,
                             "sample %s expects float32 input/output tensors with equal sizes\n",
                             sample_.name);
                return false;
            }
        }
        return true;
    }

    bool allocate_io() {
        // Allocate one CMM buffer for every model input.
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

    bool fill_inputs() {
        // Fill each input with the values declared by the selected demo.
        std::vector<float*> inputs(io_.nInputSize);
        for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
            inputs[i] = static_cast<float*>(io_.pInputs[i].pVirAddr);
        }
        sample_.fill(inputs.data(), element_count_);
        return true;
    }

    bool infer() {
        // Run inference synchronously on the NPU.
        AX_S32 ret = AX_ENGINE_RunSync(handle_, &io_);
        if (ret != 0) {
            print_error("AX_ENGINE_RunSync", ret);
            return false;
        }
        return true;
    }

    bool report() const {
        // Print the actual input and output values used by this demo.
        for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
            print_values(io_info_->pInputs[i].pName,
                         static_cast<const float*>(io_.pInputs[i].pVirAddr), element_count_);
        }
        for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
            print_values(io_info_->pOutputs[i].pName,
                         static_cast<const float*>(io_.pOutputs[i].pVirAddr), element_count_);
        }
        return true;
    }

    bool verify() const {
        // Expose the tensors to the sample-specific CPU reference.
        std::vector<const float*> inputs(io_.nInputSize);
        for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
            inputs[i] = static_cast<const float*>(io_.pInputs[i].pVirAddr);
        }

        const SampleContext context{
            inputs.data(),
            static_cast<const float*>(io_.pOutputs[0].pVirAddr),
            element_count_,
            io_info_->pOutputs[0],
        };
        float max_error = 0.0f;
        if (!sample_.verify(context, max_error)) {
            std::fprintf(stderr, "sample %s verification failed\n", sample_.name);
            return false;
        }
        if (max_error > kMaxError) {
            std::fprintf(stderr, "verification failed, max error: %g\n", max_error);
            return false;
        }

        std::printf("AX Engine %s sample: PASS (%zu elements, max error %.6g)\n", sample_.name,
                    element_count_, max_error);
        return true;
    }

    bool allocate_buffer(AX_ENGINE_IO_BUFFER_T* buffer, const AX_ENGINE_IOMETA_T& meta) {
        // Allocate a physically contiguous CMM buffer for one tensor.
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

    void release() {
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

    const SampleDefinition& sample_;
    std::vector<char> model_buffer_;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* io_info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    std::size_t element_count_ = 0;
    bool sys_initialized_ = false;
    bool engine_initialized_ = false;
};

} // namespace

bool has_shape(const AX_ENGINE_IOMETA_T& meta, std::initializer_list<AX_U32> expected_shape) {
    if (meta.nShapeSize != expected_shape.size()) {
        return false;
    }
    auto expected = expected_shape.begin();
    for (AX_U8 i = 0; i < meta.nShapeSize; ++i, ++expected) {
        if (meta.pShape[i] < 0 || static_cast<AX_U32>(meta.pShape[i]) != *expected) {
            return false;
        }
    }
    return true;
}

int run_sample_cli(int argc, char** argv, const SampleDefinition& sample) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <model.axmodel>\n", argv[0]);
        return 1;
    }

    AxEngineRunner runner(sample);
    return runner.run(argv[1]) ? 0 : 1;
}

} // namespace ax_dsl
