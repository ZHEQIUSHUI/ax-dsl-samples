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
constexpr const char* kCmmToken = "ax_dsl_runner";
constexpr float kMaxError = 1e-4f;

enum class Op {
    Add,
    Mul,
    Gelu,
    Sigmoid,
    Softmax,
};

bool parse_op(const char* text, Op& op) {
    if (std::strcmp(text, "add") == 0) {
        op = Op::Add;
    } else if (std::strcmp(text, "mul") == 0) {
        op = Op::Mul;
    } else if (std::strcmp(text, "gelu") == 0) {
        op = Op::Gelu;
    } else if (std::strcmp(text, "sigmoid") == 0) {
        op = Op::Sigmoid;
    } else if (std::strcmp(text, "softmax") == 0) {
        op = Op::Softmax;
    } else {
        return false;
    }
    return true;
}

const char* op_name(Op op) {
    switch (op) {
    case Op::Add:
        return "add";
    case Op::Mul:
        return "mul";
    case Op::Gelu:
        return "gelu";
    case Op::Sigmoid:
        return "sigmoid";
    case Op::Softmax:
        return "softmax";
    }
    return "unknown";
}

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

void print_usage(const char* program) {
    std::fprintf(stderr, "Usage: %s --op <add|mul|gelu|sigmoid|softmax> <model.axmodel>\n", program);
}

class AxDslRunner {
public:
    explicit AxDslRunner(Op op) : op_(op) {}
    ~AxDslRunner() { release(); }

    AxDslRunner(const AxDslRunner&) = delete;
    AxDslRunner& operator=(const AxDslRunner&) = delete;

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
        std::printf("sample: %s\n", op_name(op_));
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

        // Validate the IO contract used by the selected sample.
        const bool binary = op_ == Op::Add || op_ == Op::Mul;
        const AX_U32 expected_inputs = binary ? 2 : 1;
        if (io_info_->nInputSize != expected_inputs || io_info_->nOutputSize != 1) {
            std::fprintf(stderr, "sample %s expects %u input(s) and 1 output\n", op_name(op_), expected_inputs);
            return false;
        }

        const AX_ENGINE_IOMETA_T& meta_out = io_info_->pOutputs[0];
        if (meta_out.eDataType != AX_ENGINE_DT_FLOAT32 || meta_out.nSize % sizeof(float) != 0) {
            std::fprintf(stderr, "sample %s expects a float32 output tensor\n", op_name(op_));
            return false;
        }
        element_count_ = meta_out.nSize / sizeof(float);

        for (AX_U32 i = 0; i < io_info_->nInputSize; ++i) {
            const AX_ENGINE_IOMETA_T& meta_in = io_info_->pInputs[i];
            if (meta_in.eDataType != AX_ENGINE_DT_FLOAT32 || meta_in.nSize != meta_out.nSize) {
                std::fprintf(stderr, "sample %s expects float32 input/output tensors with equal sizes\n",
                             op_name(op_));
                return false;
            }
        }

        if (op_ == Op::Softmax) {
            if (meta_out.nShapeSize == 0) {
                std::fprintf(stderr, "softmax requires a known output shape\n");
                return false;
            }
            last_dim_ = static_cast<size_t>(meta_out.pShape[meta_out.nShapeSize - 1]);
            if (last_dim_ == 0 || element_count_ % last_dim_ != 0) {
                std::fprintf(stderr, "invalid softmax shape\n");
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
        // Fill each input with deterministic values.
        for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
            float* data = static_cast<float*>(io_.pInputs[i].pVirAddr);
            const float scale = static_cast<float>(i + 1) * 0.001f;
            for (size_t j = 0; j < element_count_; ++j) {
                data[j] = static_cast<float>(j % 1024) * scale;
            }
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
        // Compare the NPU output with a CPU reference for the selected sample.
        const float* output = static_cast<const float*>(io_.pOutputs[0].pVirAddr);
        float max_error = 0.0f;

        if (op_ == Op::Add || op_ == Op::Mul) {
            const float* a = static_cast<const float*>(io_.pInputs[0].pVirAddr);
            const float* b = static_cast<const float*>(io_.pInputs[1].pVirAddr);
            for (size_t i = 0; i < element_count_; ++i) {
                const float expected = op_ == Op::Add ? a[i] + b[i] : a[i] * b[i];
                max_error = std::max(max_error, std::fabs(output[i] - expected));
            }
        } else if (op_ == Op::Gelu || op_ == Op::Sigmoid) {
            const float* x = static_cast<const float*>(io_.pInputs[0].pVirAddr);
            constexpr double kInvSqrt2 = 0.70710678118654752440;
            for (size_t i = 0; i < element_count_; ++i) {
                const double value = static_cast<double>(x[i]);
                const double expected = op_ == Op::Gelu
                                            ? 0.5 * value * (1.0 + std::erf(value * kInvSqrt2))
                                            : 1.0 / (1.0 + std::exp(-value));
                max_error = std::max(max_error, std::fabs(output[i] - static_cast<float>(expected)));
            }
        } else {
            const float* x = static_cast<const float*>(io_.pInputs[0].pVirAddr);
            const size_t rows = element_count_ / last_dim_;
            for (size_t row = 0; row < rows; ++row) {
                const float* values = x + row * last_dim_;
                const float row_max = *std::max_element(values, values + last_dim_);
                double sum = 0.0;
                for (size_t col = 0; col < last_dim_; ++col) {
                    sum += std::exp(static_cast<double>(values[col] - row_max));
                }
                for (size_t col = 0; col < last_dim_; ++col) {
                    const double expected = std::exp(static_cast<double>(values[col] - row_max)) / sum;
                    max_error = std::max(max_error, std::fabs(output[row * last_dim_ + col] -
                                                              static_cast<float>(expected)));
                }
            }
        }

        if (max_error > kMaxError) {
            std::fprintf(stderr, "verification failed, max error: %g\n", max_error);
            return false;
        }

        std::printf("AX Engine %s sample: PASS (%zu elements, max error %.6g)\n", op_name(op_),
                    element_count_, max_error);
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

    Op op_;
    std::vector<char> model_buffer_;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* io_info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    size_t element_count_ = 0;
    size_t last_dim_ = 0;
    bool sys_initialized_ = false;
    bool engine_initialized_ = false;
};

} // namespace

int main(int argc, char** argv) {
    const char* op_text = nullptr;
    const char* model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
            op_text = argv[++i];
        } else if (model_path == nullptr && argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (op_text == nullptr || model_path == nullptr) {
        print_usage(argv[0]);
        return 1;
    }

    Op op;
    if (!parse_op(op_text, op)) {
        std::fprintf(stderr, "unsupported op: %s\n", op_text);
        print_usage(argv[0]);
        return 1;
    }

    AxDslRunner runner(op);
    if (!runner.init_system() || !runner.load_model(model_path) || !runner.allocate_io() ||
        !runner.fill_inputs() || !runner.infer() || !runner.verify()) {
        return 1;
    }
    return 0;
}
