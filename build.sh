#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CXX="${CXX:-aarch64-none-linux-gnu-g++}"
SAMPLES=(add mul gelu sigmoid softmax)

if [[ -z "${AX8860_BSP:-}" ]]; then
    echo "Set AX8860_BSP to the AX8860 arm64_glibc BSP directory" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

for sample in "${SAMPLES[@]}"; do
    "${CXX}" \
        -std=c++17 -O2 -Wall -Wextra -Werror \
        -I"${ROOT_DIR}/src" \
        -I"${AX8860_BSP}/include" \
        "${ROOT_DIR}/src/ax_dsl_runner.cc" \
        "${ROOT_DIR}/src/samples/${sample}.cc" \
        -L"${AX8860_BSP}/lib" -Wl,-rpath-link,"${AX8860_BSP}/lib" \
        -lax_engine -lax_sys -lpthread \
        -o "${BUILD_DIR}/ax_dsl_${sample}"
    echo "built ${BUILD_DIR}/ax_dsl_${sample}"
done
