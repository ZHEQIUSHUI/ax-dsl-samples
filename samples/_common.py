from pathlib import Path

import axnn.dsl as ax


def export_kernel(kernel, output_path, args, grid_shape, core_num=1):
    """Compile a sample kernel and store it under the requested AXModel path."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kernel.dump_path = output_path.parent

    ax.init(chip_num=1, core_num=1)
    compiled_path = kernel[grid_shape, core_num].compile(*args)
    if compiled_path.resolve() != output_path:
        compiled_path.replace(output_path)
    return output_path
