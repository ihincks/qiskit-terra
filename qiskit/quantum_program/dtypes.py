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

"""Dtype sugar for declaring :class:`.QuantumProgram` inputs.

Each dtype object supports ``[]`` indexing to build a :class:`.TensorSpec` with that
dtype and the given shape, e.g. ``f64[3]`` (a length-3 vector of ``f64``), ``f64[3, 4]``
(a 3x4 matrix), or ``f64[()]`` (a scalar). Shape entries may also be strings, which
declare a named (symbolic) dimension that is carried through type inference unresolved,
e.g. ``f64["n"]``.
"""

from qiskit._accelerate.quantum_program import TensorSpec


class _DTypeSugar:
    """A dtype that can be indexed with a shape to produce a :class:`.TensorSpec`."""

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def __getitem__(self, shape) -> TensorSpec:
        if not isinstance(shape, tuple):
            shape = (shape,)
        return TensorSpec(self._name, list(shape))

    def __repr__(self) -> str:
        return self._name


c128 = _DTypeSugar("c128")
"""128-bit complex (two ``f64`` components)."""

c64 = _DTypeSugar("c64")
"""64-bit complex (two ``f32`` components)."""

f64 = _DTypeSugar("f64")
"""64-bit floating point."""

f32 = _DTypeSugar("f32")
"""32-bit floating point."""

i64 = _DTypeSugar("i64")
"""64-bit signed integer."""

i32 = _DTypeSugar("i32")
"""32-bit signed integer."""

i16 = _DTypeSugar("i16")
"""16-bit signed integer."""

i8 = _DTypeSugar("i8")
"""8-bit signed integer."""

u64 = _DTypeSugar("u64")
"""64-bit unsigned integer."""

u32 = _DTypeSugar("u32")
"""32-bit unsigned integer."""

u16 = _DTypeSugar("u16")
"""16-bit unsigned integer."""

u8 = _DTypeSugar("u8")
"""8-bit unsigned integer."""

bit = _DTypeSugar("bit")
"""A single classical bit."""
