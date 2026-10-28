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
        std::printf("  [%u] %s %s ", i, meta.pName, dtype_name(meta.eDataType));
        print_shape(meta);
        std::printf(" (%u bytes)\n", meta.nSize);
    }

    std::printf("outputs: %u\n", info.nOutputSize);
    for (AX_U32 i = 0; i < info.nOutputSize; ++i) {
        const AX_ENGINE_IOMETA_T& meta = info.pOutputs[i];
        std::printf("  [%u] %s %s ", i, meta.pName, dtype_name(meta.eDataType));
        print_shape(meta);
        std::printf(" (%u bytes)\n", meta.nSize);
    }
}

int alloc_buffer(AX_ENGINE_IO_BUFFER_T* buffer, const AX_ENGINE_IOMETA_T& meta) {
    std::memset(buffer, 0, sizeof(*buffer));
    buffer->nSize = meta.nSize;
    return AX_SYS_MemAlloc(
        &buffer->phyAddr,
        &buffer->pVirAddr,
        meta.nSize,
        kCmmAlignSize,
        reinterpret_cast<const AX_S8*>(kCmmToken));
}

void free_io(AX_ENGINE_IO_T* io) {
    if (io->pInputs != nullptr) {
        for (AX_U32 i = 0; i < io->nInputSize; ++i) {
            if (io->pInputs[i].pVirAddr != nullptr) {
                AX_SYS_MemFree(io->pInputs[i].phyAddr, io->pInputs[i].pVirAddr);
            }
        }
        delete[] io->pInputs;
        io->pInputs = nullptr;
    }

    if (io->pOutputs != nullptr) {
        for (AX_U32 i = 0; i < io->nOutputSize; ++i) {
            if (io->pOutputs[i].pVirAddr != nullptr) {
                AX_SYS_MemFree(io->pOutputs[i].phyAddr, io->pOutputs[i].pVirAddr);
            }
        }
        delete[] io->pOutputs;
        io->pOutputs = nullptr;
    }
}

void print_error(const char* api, AX_S32 ret) {
    std::fprintf(stderr, "%s failed: 0x%08x\n", api, static_cast<unsigned>(ret));
}

} // namespace

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "models/demo_add_ax8860.axmodel";
    std::vector<char> model_buffer;
    if (!read_file(model_path, model_buffer)) {
        std::fprintf(stderr, "failed to read AXModel: %s\n", model_path);
        return 1;
    }
    if (model_buffer.size() > std::numeric_limits<AX_U32>::max()) {
        std::fprintf(stderr, "AXModel is too large: %zu bytes\n", model_buffer.size());
        return 1;
    }

    bool sys_initialized = false;
    bool engine_initialized = false;
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_IO_INFO_T* io_info = nullptr;
    AX_ENGINE_IO_T io{};
    int result = 1;

    AX_S32 ret = AX_SYS_Init();
    if (ret != 0) {
        print_error("AX_SYS_Init", ret);
        goto cleanup;
    }
    sys_initialized = true;

    AX_ENGINE_NPU_ATTR_T npu_attr;
    std::memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        print_error("AX_ENGINE_Init", ret);
        goto cleanup;
    }
    engine_initialized = true;

    std::printf("AX Engine version: %s\n", AX_ENGINE_GetVersion());
    ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), static_cast<AX_U32>(model_buffer.size()));
    if (ret != 0) {
        print_error("AX_ENGINE_CreateHandle", ret);
        goto cleanup;
    }

    ret = AX_ENGINE_CreateContext(handle);
    if (ret != 0) {
        print_error("AX_ENGINE_CreateContext", ret);
        goto cleanup;
    }

    ret = AX_ENGINE_GetIOInfo(handle, &io_info);
    if (ret != 0) {
        print_error("AX_ENGINE_GetIOInfo", ret);
        goto cleanup;
    }
    print_io_info(*io_info);

    {
        const int input_a = find_io(io_info->pInputs, io_info->nInputSize, "a");
        const int input_b = find_io(io_info->pInputs, io_info->nInputSize, "b");
        const int output = find_io(io_info->pOutputs, io_info->nOutputSize, "output");
        if (input_a < 0 || input_b < 0 || output < 0) {
            std::fprintf(stderr, "expected inputs a, b and output output in AXModel\n");
            goto cleanup;
        }

        const AX_ENGINE_IOMETA_T& meta_a = io_info->pInputs[input_a];
        const AX_ENGINE_IOMETA_T& meta_b = io_info->pInputs[input_b];
        const AX_ENGINE_IOMETA_T& meta_out = io_info->pOutputs[output];
        if (meta_a.eDataType != AX_ENGINE_DT_FLOAT32 || meta_b.eDataType != AX_ENGINE_DT_FLOAT32 ||
            meta_out.eDataType != AX_ENGINE_DT_FLOAT32) {
            std::fprintf(stderr, "this demo expects float32 input and output tensors\n");
            goto cleanup;
        }
        if (meta_a.nSize != meta_b.nSize || meta_a.nSize != meta_out.nSize ||
            meta_a.nSize % sizeof(float) != 0) {
            std::fprintf(stderr, "unexpected tensor sizes: a=%u b=%u output=%u\n", meta_a.nSize, meta_b.nSize,
                         meta_out.nSize);
            goto cleanup;
        }

        io.nInputSize = io_info->nInputSize;
        io.pInputs = new AX_ENGINE_IO_BUFFER_T[io.nInputSize]{};
        for (AX_U32 i = 0; i < io.nInputSize; ++i) {
            ret = alloc_buffer(&io.pInputs[i], io_info->pInputs[i]);
            if (ret != 0) {
                print_error("AX_SYS_MemAlloc(input)", ret);
                goto cleanup;
            }
        }

        io.nOutputSize = io_info->nOutputSize;
        io.pOutputs = new AX_ENGINE_IO_BUFFER_T[io.nOutputSize]{};
        for (AX_U32 i = 0; i < io.nOutputSize; ++i) {
            ret = alloc_buffer(&io.pOutputs[i], io_info->pOutputs[i]);
            if (ret != 0) {
                print_error("AX_SYS_MemAlloc(output)", ret);
                goto cleanup;
            }
        }

        const size_t element_count = meta_a.nSize / sizeof(float);
        float* a = static_cast<float*>(io.pInputs[input_a].pVirAddr);
        float* b = static_cast<float*>(io.pInputs[input_b].pVirAddr);
        for (size_t i = 0; i < element_count; ++i) {
            a[i] = static_cast<float>(i % 1024) * 0.001f;
            b[i] = static_cast<float>(i % 1024) * 0.002f;
        }

        ret = AX_ENGINE_RunSync(handle, &io);
        if (ret != 0) {
            print_error("AX_ENGINE_RunSync", ret);
            goto cleanup;
        }

        const float* output_data = static_cast<const float*>(io.pOutputs[output].pVirAddr);
        float max_error = 0.0f;
        for (size_t i = 0; i < element_count; ++i) {
            const float expected = a[i] + b[i];
            max_error = std::max(max_error, std::fabs(output_data[i] - expected));
        }
        if (max_error > 1e-5f) {
            std::fprintf(stderr, "verification failed, max error: %g\n", max_error);
            goto cleanup;
        }

        std::printf("AX Engine demo add: PASS (%zu elements, max error %.6g)\n", element_count, max_error);
        result = 0;
    }

cleanup:
    free_io(&io);
    if (handle != nullptr) {
        AX_ENGINE_DestroyHandle(handle);
    }
    if (engine_initialized) {
        AX_ENGINE_Deinit();
    }
    if (sys_initialized) {
        AX_SYS_Deinit();
    }
    return result;
}

