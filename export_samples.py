#!/usr/bin/env python3
"""Export all AXDSL samples as AX8860 AXModels."""

import argparse
from pathlib import Path

from samples import add, gelu, mul, sigmoid, softmax


SAMPLE_MODULES = (
    ("add", add),
    ("mul", mul),
    ("gelu", gelu),
    ("sigmoid", sigmoid),
    ("softmax", softmax),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="directory for generated AXModels",
    )
    args = parser.parse_args()

    for name, module in SAMPLE_MODULES:
        model_path = module.export_model(
            args.output_dir / f"demo_{name}_ax8860.axmodel"
        )
        print(f"exported {name}: {model_path}")


if __name__ == "__main__":
    main()
