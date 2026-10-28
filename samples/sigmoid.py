"""AXDSL sigmoid sample."""

import torch

import axnn.dsl as ax
from samples._common import export_kernel


TILE_SIZE = [128]
SHAPE = (256, 256)


@ax.builder(run_mode=ax.RunMode.PYRUN)
def sigmoid_kernel(x, output, tile_size):
    offset = ax.program_id(0) * tile_size[0]
    tile = (slice(offset, offset + tile_size[0]), slice(0, None))

    x_tile = ax.dsl.load(x[tile])
    result = ax.dsl.sigmoid(x_tile)
    ax.dsl.store(x=result, r=output[tile])
    return output


def run(x, output):
    grid = (SHAPE[0] // TILE_SIZE[0],)
    return sigmoid_kernel[grid, 1](x, output, TILE_SIZE)


def export_model(output_path):
    x = torch.empty(SHAPE, dtype=torch.float32)
    output = torch.empty(SHAPE, dtype=torch.float32)
    grid = (SHAPE[0] // TILE_SIZE[0],)
    return export_kernel(sigmoid_kernel, output_path, (x, output, TILE_SIZE), grid)


def main():
    ax.init(chip_num=1, core_num=1)
    x = torch.rand(SHAPE, dtype=torch.float32)
    output = torch.empty_like(x)
    run(x, output)
    torch.testing.assert_close(output, torch.sigmoid(x), atol=1e-5, rtol=1e-5)
    print("AXDSL sigmoid sample: PASS")


if __name__ == "__main__":
    main()
