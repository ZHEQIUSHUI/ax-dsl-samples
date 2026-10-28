"""AXDSL GELU demo: apply GELU to a vector."""

import torch

import axnn.dsl as ax
from samples._common import export_kernel


SHAPE = (8,)
X_VALUES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)


@ax.builder(run_mode=ax.RunMode.PYRUN)
def gelu_kernel(x, output):
    x_values = ax.dsl.load(x)
    result = ax.dsl.gelu(x_values)
    ax.dsl.store(x=result, r=output)
    return output


def run(x, output):
    return gelu_kernel[(1,), 1](x, output)


def export_model(output_path):
    x = torch.empty(SHAPE, dtype=torch.float32)
    output = torch.empty(SHAPE, dtype=torch.float32)
    return export_kernel(gelu_kernel, output_path, (x, output), (1,))


def main():
    ax.init(chip_num=1, core_num=1)
    x = torch.tensor(X_VALUES, dtype=torch.float32)
    output = torch.empty_like(x)
    run(x, output)

    print("x        =", x.tolist())
    print("gelu(x)  =", output.tolist())
    torch.testing.assert_close(
        output, torch.nn.functional.gelu(x), atol=1e-5, rtol=1e-5
    )
    print("AXDSL gelu sample: PASS")


if __name__ == "__main__":
    main()
