#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include <ax_engine_api.h>
#include <ax_sys_api.h>

namespace {

constexpr AX_U32 kCmmAlignSize = 128;
constexpr const char* kCmmToken = "ax_demo_add";

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
    data.resize(static_cast<size_t>(size));
    return static_cast<bool>(file.read(data.data(), size));
}

int find_io(const AX_ENGINE_IOMETA_T* metas, AX_U32 count, const char* name) {
    for (AX_U32 i = 0; i < count; ++i) {
        if (metas[i].pName != nullptr && std::strcmp(metas[i].pName, name) == 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
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

class AxDemo {
public:
    AxDemo() = default;
    ~AxDemo() { release(); }

    AxDemo(const AxDemo&) = delete;
    AxDemo& operator=(const AxDemo&) = delete;

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

        // Locate the two inputs and one output used by this demo.
        input_a_ = find_io(io_info_->pInputs, io_info_->nInputSize, "a");
        input_b_ = find_io(io_info_->pInputs, io_info_->nInputSize, "b");
        output_ = find_io(io_info_->pOutputs, io_info_->nOutputSize, "output");
        if (input_a_ < 0 || input_b_ < 0 || output_ < 0) {
            std::fprintf(stderr, "expected inputs a, b and output output in AXModel\n");
            return false;
        }

        const AX_ENGINE_IOMETA_T& meta_a = io_info_->pInputs[input_a_];
        const AX_ENGINE_IOMETA_T& meta_b = io_info_->pInputs[input_b_];
        const AX_ENGINE_IOMETA_T& meta_out = io_info_->pOutputs[output_];
        if (meta_a.eDataType != AX_ENGINE_DT_FLOAT32 || meta_b.eDataType != AX_ENGINE_DT_FLOAT32 ||
            meta_out.eDataType != AX_ENGINE_DT_FLOAT32) {
            std::fprintf(stderr, "this demo expects float32 input and output tensors\n");
            return false;
        }
        if (meta_a.nSize != meta_b.nSize || meta_a.nSize != meta_out.nSize ||
            meta_a.nSize % sizeof(float) != 0) {
            std::fprintf(stderr, "unexpected tensor sizes: a=%u b=%u output=%u\n", meta_a.nSize, meta_b.nSize,
                         meta_out.nSize);
            return false;
        }

        element_count_ = meta_a.nSize / sizeof(float);
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
        // Fill the CMM input buffers with deterministic test data.
        float* a = static_cast<float*>(io_.pInputs[input_a_].pVirAddr);
        float* b = static_cast<float*>(io_.pInputs[input_b_].pVirAddr);
        for (size_t i = 0; i < element_count_; ++i) {
            a[i] = static_cast<float>(i % 1024) * 0.001f;
            b[i] = static_cast<float>(i % 1024) * 0.002f;
        }
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

    bool verify() const {
        // Compare the NPU output with the CPU reference a + b.
        const float* a = static_cast<const float*>(io_.pInputs[input_a_].pVirAddr);
        const float* b = static_cast<const float*>(io_.pInputs[input_b_].pVirAddr);
        const float* output = static_cast<const float*>(io_.pOutputs[output_].pVirAddr);

        float max_error = 0.0f;
        for (size_t i = 0; i < element_count_; ++i) {
            max_error = std::max(max_error, std::fabs(output[i] - (a[i] + b[i])));
        }
        if (max_error > 1e-5f) {
            std::fprintf(stderr, "verification failed, max error: %g\n", max_error);
            return false;
        }

        std::printf("AX Engine demo add: PASS (%zu elements, max error %.6g)\n", element_count_, max_error);
        return true;
    }

private:
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

    std::vector<char> model_buffer_;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* io_info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    int input_a_ = -1;
    int input_b_ = -1;
    int output_ = -1;
    size_t element_count_ = 0;
    bool sys_initialized_ = false;
    bool engine_initialized_ = false;
};

} // namespace

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "models/demo_add_ax8860.axmodel";

    AxDemo demo;
    if (!demo.init_system() || !demo.load_model(model_path) || !demo.allocate_io() || !demo.fill_inputs() ||
        !demo.infer() || !demo.verify()) {
        return 1;
    }
    return 0;
}
