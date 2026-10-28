#!/usr/bin/env python3
"""Export the AXDSL demo_add kernel as an AX8860 AXModel."""

import argparse
from pathlib import Path

import torch

import axnn.dsl as ax
from axnn.dsl.demo.single_op_demo.demo_add import ax_custom_op_kernel


TILE_SIZE = [128, 256]
INPUT_SHAPE = (256, 512)


def export_axmodel(output_path):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax_custom_op_kernel.dump_path = output_path.parent
    ax.init(chip_num=1, core_num=1)
    a = torch.empty(INPUT_SHAPE, dtype=torch.float32)
    b = torch.empty(INPUT_SHAPE, dtype=torch.float32)
    output = torch.empty(INPUT_SHAPE, dtype=torch.float32)
    grid = (INPUT_SHAPE[0] // TILE_SIZE[0], INPUT_SHAPE[1] // TILE_SIZE[1])

    compiled_path = ax_custom_op_kernel[grid, 1].compile(a, b, output, TILE_SIZE)
    if compiled_path.resolve() != output_path:
        compiled_path.replace(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "demo_add_ax8860.axmodel",
        help="output AXModel path",
    )
    args = parser.parse_args()

    model_path = export_axmodel(args.output)
    print(f"exported AX8860 AXModel: {model_path}")


if __name__ == "__main__":
    main()
