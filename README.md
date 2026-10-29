# AXDSL Samples for AX8860

This project contains small AXDSL samples, exports them as AX8860 AXModels, and runs each one with
its own AX Engine C++ executable. The lifecycle code is shared, while model-specific verification
stays in one source file per sample. It does not use AXCL.

## Samples

Each sample is a worked example rather than a benchmark: it uses a tiny fixed input, applies one
AXDSL operation, prints the result, and checks it against a CPU calculation. For example, `add`
takes two eight-element vectors and produces `[11, 22, 33, 44, 55, 66, 77, 88]`.

Every sample uses a small, readable float32 vector with 8 elements:

```text
A = [1, 2, 3, 4, 5, 6, 7, 8]
B = [10, 20, 30, 40, 50, 60, 70, 80]
X = [-3, -2, -1, 0, 1, 2, 3, 4]
```

| Sample | Demonstration |
| ------ | ------------- |
| `add` | `A + B` |
| `mul` | `A * B` |
| `gelu` | `GELU(X)` |
| `sigmoid` | `sigmoid(X)` |
| `softmax` | Convert the scores in `X` into probabilities |

The PYRUN programs and the board executables print these inputs and the actual output values, so
the operation can be understood without inspecting generated model data.

The generated AXModels are stored in `models/` and can be used directly. They are also reproducible
with `export_samples.py`.

## Layout

```text
ax-dsl-samples/
├── .github/workflows/build.yml
├── export_samples.py
├── samples/
│   ├── add.py
│   ├── mul.py
│   ├── gelu.py
│   ├── sigmoid.py
│   └── softmax.py
├── src/
│   ├── ax_dsl_runner.h
│   ├── ax_dsl_runner.cc
│   └── samples/
│       ├── add.cc
│       ├── mul.cc
│       ├── gelu.cc
│       ├── sigmoid.cc
│       └── softmax.cc
├── models/demo_*_ax8860.axmodel
├── CMakeLists.txt
├── cmake/aarch64-none-linux-gnu.toolchain.cmake
└── build.sh
```

Each Python sample module defines a PYRUN kernel for local correctness checking and an
`export_model()` function used by the exporter. Each C++ sample has its own `main()` with the full
board-side flow from initialization through printing the result.

## Prerequisites

- NPU codebase environment for importing `axnn.dsl`
- AX8860 BSP with `include/ax_engine_api.h`, `lib/libax_engine.so`, `lib/libax_sys.so`, and
  `lib/libax_interpreter.so`
- Arm GNU Toolchain 13.3.rel1 or another aarch64 toolchain with glibc >= 2.38

## Run A Sample With PYRUN

```bash
export NPU_CODEBASE=/path/to/npu-codebase
source "$NPU_CODEBASE/script/npu_dev"
python3 -m samples.add
python3 -m samples.mul
python3 -m samples.gelu
python3 -m samples.sigmoid
python3 -m samples.softmax
```

## Export All AXModels

```bash
export NPU_CODEBASE=/path/to/npu-codebase
source "$NPU_CODEBASE/script/npu_dev"
python3 export_samples.py
```

The default output directory is `models/`. Each sample produces
`models/demo_<sample>_ax8860.axmodel`.

## Cross Compile

Set the BSP directory and put the toolchain on `PATH`:

```bash
export AX8860_BSP=/path/to/ax8860_bsp/arm64_glibc
export PATH=/path/to/arm-gnu-toolchain-13.3.rel1-x86_64-aarch64-none-linux-gnu/bin:$PATH
./build.sh
```

The executables are written to `build/ax_dsl_<sample>`:

```text
build/ax_dsl_add
build/ax_dsl_mul
build/ax_dsl_gelu
build/ax_dsl_sigmoid
build/ax_dsl_softmax
```

CMake is also supported:

```bash
cmake \
  -S . \
  -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-none-linux-gnu.toolchain.cmake \
  -DAX8860_BSP="${AX8860_BSP}"
cmake --build build -j
```

The CI toolchain package for an x86_64 Linux host is:

```text
https://gitlab.arm.com/api/v4/projects/tooling%2Fgnu-toolchains-for-arm/packages/generic/gnu-toolchain/13.3.rel1/arm-gnu-toolchain-13.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
```

Its SHA256 is:

```text
322f0b4482fc0d9fa0bb468134841f08d8c554c54ff5aa29a13a7a24bf7e1eb5
```

## Continuous Integration

`.github/workflows/build.yml` downloads the pinned Arm GNU Toolchain and the minimal AX8860 BSP,
builds every sample with both `build.sh` and CMake, and verifies the resulting executables. It does
not upload build artifacts.

The workflow downloads the BSP bundle from this release asset:

```text
https://github.com/ZHEQIUSHUI/ax-dsl-samples/releases/download/bsp/ax8860_arm64_glibc_axengine_dev.zip
```

The expected bundle SHA256 is
`ebbbb66065a57bdbb52d58cea93b32986dcfc4601c262161f7051a40dae00b3e`.

## Run On The Board

Copy the executables and models to the board:

```bash
ssh root@<board-ip> 'mkdir -p /tmp/ax-dsl-samples/models'
scp build/ax_dsl_* root@<board-ip>:/tmp/ax-dsl-samples/
scp models/*.axmodel root@<board-ip>:/tmp/ax-dsl-samples/models/
ssh root@<board-ip>
cd /tmp/ax-dsl-samples
export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
```

Run each sample with its matching executable:

```bash
./ax_dsl_add models/demo_add_ax8860.axmodel
./ax_dsl_mul models/demo_mul_ax8860.axmodel
./ax_dsl_gelu models/demo_gelu_ax8860.axmodel
./ax_dsl_sigmoid models/demo_sigmoid_ax8860.axmodel
./ax_dsl_softmax models/demo_softmax_ax8860.axmodel
```

The executable prints the model metadata, the input vectors, and the output vector:

```text
AX Engine version: 3.0.001
inputs: 2
  [0] a FLOAT32 8 (32 bytes)
  [1] b FLOAT32 8 (32 bytes)
outputs: 1
  [0] output FLOAT32 8 (32 bytes)
a: [1, 2, 3, 4, 5, 6, 7, 8]
b: [10, 20, 30, 40, 50, 60, 70, 80]
a + b: [11, 22, 33, 44, 55, 66, 77, 88]
AX Engine add sample: PASS (8 elements, max error 0)
```

## C++ Runner Structure

`src/ax_dsl_runner.cc` is a small wrapper around the AX Engine APIs. Each sample's `main()` calls
the wrapper directly, in this order:

- `init()`: `AX_SYS_Init` for system/CMM services, then `AX_ENGINE_Init` for the NPU.
- `load_model()`: read the AXModel, create the handle/context, and query the IO metadata.
- `allocate_io()`: allocate CMM buffers for every model input and output.
- `input(i)`: return the writable buffer for model input `i`, so the sample can copy its data in.
- `run()`: call `AX_ENGINE_RunSync`.
- `output(i)`: return the result buffer for model output `i`, which the sample prints.
- `release()`: free CMM buffers and finalize the engine/system in reverse order.

`src/samples/<sample>.cc` contains the complete example: input values, the six lifecycle steps in
`main()`, result printing, and the CPU reference check. There is no callback registration or
CLI framework between the sample and the AX Engine calls.

