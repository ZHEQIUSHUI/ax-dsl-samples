"""AXDSL elementwise multiply demo: multiply two vectors."""

import torch

import axnn.dsl as ax
from samples._common import export_kernel


SHAPE = (8,)
A_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
B_VALUES = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)


@ax.builder(run_mode=ax.RunMode.PYRUN)
def mul_kernel(a, b, output):
    a_values = ax.dsl.load(a)
    b_values = ax.dsl.load(b)
    result = ax.dsl.mul(a_values, b_values)
    ax.dsl.store(x=result, r=output)
    return output


def run(a, b, output):
    return mul_kernel[(1,), 1](a, b, output)


def export_model(output_path):
    a = torch.empty(SHAPE, dtype=torch.float32)
    b = torch.empty(SHAPE, dtype=torch.float32)
    output = torch.empty(SHAPE, dtype=torch.float32)
    return export_kernel(mul_kernel, output_path, (a, b, output), (1,))


def main():
    ax.init(chip_num=1, core_num=1)
    a = torch.tensor(A_VALUES, dtype=torch.float32)
    b = torch.tensor(B_VALUES, dtype=torch.float32)
    output = torch.empty_like(a)
    run(a, b, output)

    print("a     =", a.tolist())
    print("b     =", b.tolist())
    print("a * b =", output.tolist())
    torch.testing.assert_close(output, a * b, atol=1e-5, rtol=1e-5)
    print("AXDSL mul sample: PASS")


if __name__ == "__main__":
    main()
