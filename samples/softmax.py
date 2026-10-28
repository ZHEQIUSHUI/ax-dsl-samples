"""AXDSL softmax demo: convert a vector of scores into probabilities."""

import torch

import axnn.dsl as ax
from samples._common import export_kernel


SHAPE = (8,)
X_VALUES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
DIM = -1


@ax.builder(run_mode=ax.RunMode.PYRUN)
def softmax_kernel(x, output, dim):
    x_values = ax.dsl.load(x)
    result = ax.dsl.softmax(x_values, dim=dim)
    ax.dsl.store(x=result, r=output)
    return output


def run(x, output):
    return softmax_kernel[(1,), 1](x, output, DIM)


def export_model(output_path):
    x = torch.empty(SHAPE, dtype=torch.float32)
    output = torch.empty(SHAPE, dtype=torch.float32)
    return export_kernel(softmax_kernel, output_path, (x, output, DIM), (1,))


def main():
    ax.init(chip_num=1, core_num=1)
    x = torch.tensor(X_VALUES, dtype=torch.float32)
    output = torch.empty_like(x)
    run(x, output)

    print("x          =", x.tolist())
    print("softmax(x) =", output.tolist())
    print("sum        =", output.sum().item())
    torch.testing.assert_close(
        output, torch.nn.functional.softmax(x, dim=DIM), atol=1e-5, rtol=1e-5
    )
    print("AXDSL softmax sample: PASS")


if __name__ == "__main__":
    main()
