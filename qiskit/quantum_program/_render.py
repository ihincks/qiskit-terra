# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rendering a program: the entry point to each form, and the rasterizer for a drawing.

Both renderings are produced by the Rust crate, which is where the program is. A drawing arrives
here as Graphviz source and is rasterized by running ``dot``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.utils import optionals as _optionals

from ._tracer import Tracer, build

__all__ = ["draw", "listing"]


def listing(program: QuantumProgram | Tracer) -> str:
    """A program as a listing of every node it holds, or one value of a program being written.

    A tracer is built into a program first, so what is listed is the program it would produce: one
    output per value it covers, and only the operations those outputs reach.

    Args:
        program: The program to list, or a tracer standing for one value of one.

    Returns:
        The listing, which is also what ``str()`` of a program gives.
    """
    return _program(program).listing()


def draw(program: QuantumProgram | Tracer) -> Image:
    """Draw a program's dataflow as a graph, or that of one value of a program being written.

    A tracer is built into a program first, so what is drawn is the program it would produce.

    Args:
        program: The program to draw, or a tracer standing for one value of one.

    Returns:
        The drawing, as a ``PIL.Image.Image``.

    Raises:
        MissingOptionalLibraryError: If Graphviz or Pillow is missing.
    """
    return _program(program).draw()


def _program(program: QuantumProgram | Tracer) -> QuantumProgram:
    """``program`` itself, or the program a tracer would build."""
    return build(program) if isinstance(program, Tracer) else program


@_optionals.HAS_GRAPHVIZ.require_in_call("drawing a quantum program")
@_optionals.HAS_PIL.require_in_call("drawing a quantum program")
def _image(source: str) -> Image:
    """The image Graphviz lays ``source`` out as.

    Args:
        source: A drawing in the Graphviz language.

    Returns:
        The rasterized drawing.
    """
    import io
    import subprocess

    from PIL import Image as pil_image

    layout = subprocess.run(
        ["dot", "-T", "png"],  # noqa: S607  dot is on PATH, which HAS_GRAPHVIZ tests
        input=source.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    return pil_image.open(io.BytesIO(layout.stdout))


if TYPE_CHECKING:
    from PIL.Image import Image

    from qiskit._accelerate.quantum_program import QuantumProgram
