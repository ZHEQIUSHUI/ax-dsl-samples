"""AXDSL elementwise add sample."""

import torch

import axnn.dsl as ax
from samples._common import export_kernel


TILE_SIZE = [128, 256]
SHAPE = (256, 512)


@ax.builder(run_mode=ax.RunMode.PYRUN)
def add_kernel(a, b, output, tile_size):
    offset_0 = ax.program_id(0) * tile_size[0]
    offset_1 = ax.program_id(1) * tile_size[1]
    tile = (
        slice(offset_0, offset_0 + tile_size[0]),
        slice(offset_1, offset_1 + tile_size[1]),
    )

    a_tile = ax.dsl.load(a[tile])
    b_tile = ax.dsl.load(b[tile])
    result = ax.dsl.add(a_tile, b_tile)
    ax.dsl.store(x=result, r=output[tile])
    return output


def run(a, b, output):
    grid = (SHAPE[0] // TILE_SIZE[0], SHAPE[1] // TILE_SIZE[1])
    return add_kernel[grid, 1](a, b, output, TILE_SIZE)


def export_model(output_path):
    a = torch.empty(SHAPE, dtype=torch.float32)
    b = torch.empty(SHAPE, dtype=torch.float32)
    output = torch.empty(SHAPE, dtype=torch.float32)
    grid = (SHAPE[0] // TILE_SIZE[0], SHAPE[1] // TILE_SIZE[1])
    return export_kernel(add_kernel, output_path, (a, b, output, TILE_SIZE), grid)


def main():
    ax.init(chip_num=1, core_num=1)
    a = torch.rand(SHAPE, dtype=torch.float32)
    b = torch.rand(SHAPE, dtype=torch.float32)
    output = torch.empty_like(a)
    run(a, b, output)
    torch.testing.assert_close(output, a + b, atol=1e-5, rtol=1e-5)
    print("AXDSL add sample: PASS")


if __name__ == "__main__":
    main()
