# AXDSL Samples for AX8860

This project contains small AXDSL samples, exports them as AX8860 AXModels, and runs them with one
shared AX Engine C++ runner. It does not use AXCL.

## Samples

| Sample | Inputs | Output shape | Notes |
| ------ | ------ | ------------ | ----- |
| `add` | `a`, `b` float32 `256x512` | `256x512` | Elementwise add |
| `mul` | `a`, `b` float32 `256x512` | `256x512` | Elementwise multiply |
| `gelu` | `x` float32 `256x256` | `256x256` | GELU activation |
| `sigmoid` | `x` float32 `256x256` | `256x256` | Sigmoid activation |
| `softmax` | `x` float32 `256x256` | `256x256` | Softmax over the last dimension |

The generated AXModels are stored in `models/` and can be used directly. They are also reproducible
with `export_samples.py`.

## Layout

```text
ax-dsl-samples/
├── export_samples.py
├── samples/
│   ├── add.py
│   ├── mul.py
│   ├── gelu.py
│   ├── sigmoid.py
│   └── softmax.py
├── src/ax_dsl_runner.cc
├── models/demo_*_ax8860.axmodel
├── CMakeLists.txt
├── cmake/aarch64-none-linux-gnu.toolchain.cmake
└── build.sh
```

Each sample module defines a PYRUN kernel for local correctness checking and an `export_model()`
function used by the exporter.

## Prerequisites

- NPU codebase environment for importing `axnn.dsl`
- AX8860 BSP with `include/ax_engine_api.h`, `lib/libax_engine.so`, and `lib/libax_sys.so`
- Arm GNU Toolchain 13.2 or another aarch64 toolchain with glibc >= 2.38

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
export PATH=/path/to/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu/bin:$PATH
./build.sh
```

The executable is written to `build/ax_dsl_runner`.

CMake is also supported:

```bash
cmake \
  -S . \
  -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-none-linux-gnu.toolchain.cmake \
  -DAX8860_BSP="${AX8860_BSP}"
cmake --build build -j
```

## Run On The Board

Copy the runner and one or more models to the board:

```bash
scp build/ax_dsl_runner models/*.axmodel root@<board-ip>:/tmp/ax-dsl-samples/
ssh root@<board-ip>
cd /tmp/ax-dsl-samples
export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
```

Run a specific sample with `--op`:

```bash
./ax_dsl_runner --op add models/demo_add_ax8860.axmodel
./ax_dsl_runner --op mul models/demo_mul_ax8860.axmodel
./ax_dsl_runner --op gelu models/demo_gelu_ax8860.axmodel
./ax_dsl_runner --op sigmoid models/demo_sigmoid_ax8860.axmodel
./ax_dsl_runner --op softmax models/demo_softmax_ax8860.axmodel
```

Example output:

```text
AX Engine version: 3.0.001
sample: add
inputs: 2
  [0] a FLOAT32 256x512 (524288 bytes)
  [1] b FLOAT32 256x512 (524288 bytes)
outputs: 1
  [0] output FLOAT32 256x512 (524288 bytes)
AX Engine add sample: PASS (131072 elements, max error 0)
```

## C++ Runner Structure

`src/ax_dsl_runner.cc` keeps the AX Engine lifecycle in small, explicit steps:

- `init_system()`: `AX_SYS_Init` for system/CMM services, then `AX_ENGINE_Init` for the NPU.
- `load_model()`: read the AXModel, create the handle/context, query and validate IO metadata.
- `allocate_io()`: allocate CMM buffers for every model input and output.
- `fill_inputs()`: fill deterministic input data.
- `infer()`: run `AX_ENGINE_RunSync`.
- `verify()`: compare the NPU output against a CPU reference selected by `--op`.
- `release()`: free CMM buffers and finalize the engine/system in reverse order.

