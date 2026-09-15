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

# The classes here are declared by hand because they come from the compiled extension module,
# which has no signatures of its own. Each member's summary line is repeated so that a hover
# keeps showing one; the full documentation lives on the Rust type. Nothing may be declared here
# that the class does not have, which test_quantum_program.py checks. A class declares only its own
# members and inherits the rest, and every type it names is imported absolutely, so that the test
# reads no import here as an export.

from typing import TYPE_CHECKING

from qiskit.quantum_program import DataTree, DType, TensorType, bounded

__all__ = [
    "Add",
    "BindParameters",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "BroadcastTo",
    "Cast",
    "Constant",
    "Divide",
    "Mean",
    "Multiply",
    "Parity",
    "Power",
    "ProgramOp",
    "Remainder",
    "ShotLoop",
    "Std",
    "Subtract",
    "Variance",
]

class ProgramOp:
    """One of the operations an instruction can perform."""

    @property
    def full_name(self) -> str:
        """The type name, qualified by its namespace."""

    @property
    def describe(self) -> str | None:
        """A summary of the payload, such as `axis=0`, and `None` for an operation with none."""

    def output_types(self, operands: Sequence[TensorType], /) -> DataTree:
        """Return the types this operation produces from operands of type `operands`."""

class Add(ProgramOp):
    """Add two tensors elementwise."""

    def __init__(self) -> None: ...

class Subtract(ProgramOp):
    """Subtract the second tensor from the first, elementwise."""

    def __init__(self) -> None: ...

class Multiply(ProgramOp):
    """Multiply two tensors elementwise."""

    def __init__(self) -> None: ...

class Divide(ProgramOp):
    """Divide the first tensor by the second, elementwise."""

    def __init__(self) -> None: ...

class Remainder(ProgramOp):
    """Remainder of the first tensor divided by the second, elementwise."""

    def __init__(self) -> None: ...

class Power(ProgramOp):
    """Raise the first tensor to the power of the second, elementwise."""

    def __init__(self) -> None: ...

class BitwiseAnd(ProgramOp):
    """Bitwise AND of two bit-valued tensors, elementwise."""

    def __init__(self) -> None: ...

class BitwiseOr(ProgramOp):
    """Bitwise OR of two bit-valued tensors, elementwise."""

    def __init__(self) -> None: ...

class BitwiseXor(ProgramOp):
    """Bitwise XOR of two bit-valued tensors, elementwise."""

    def __init__(self) -> None: ...

class BitwiseNot(ProgramOp):
    """Bitwise NOT of a bit-valued tensor, elementwise."""

    def __init__(self) -> None: ...

class Parity(ProgramOp):
    """XOR-reduce the bits of a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /) -> None: ...
    @property
    def axis(self) -> int:
        """The axis this XOR-reduces along."""

class Mean(ProgramOp):
    """Average a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /) -> None: ...
    @property
    def axis(self) -> int:
        """The axis this averages along."""

class Variance(ProgramOp):
    """Variance of a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /, ddof: float = 0.0) -> None: ...
    @property
    def axis(self) -> int:
        """The axis this takes the variance along."""

    @property
    def ddof(self) -> float:
        """The delta degrees of freedom, subtracted from the divisor."""

class Std(ProgramOp):
    """Standard deviation of a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /, ddof: float = 0.0) -> None: ...
    @property
    def axis(self) -> int:
        """The axis this takes the standard deviation along."""

    @property
    def ddof(self) -> float:
        """The delta degrees of freedom, subtracted from the divisor."""

class Cast(ProgramOp):
    """Reinterpret a tensor as another dtype, keeping its shape."""

    def __init__(self, target: DType, /) -> None: ...
    @property
    def target(self) -> DType:
        """The dtype this casts to."""

class BroadcastTo(ProgramOp):
    """Broadcast a tensor to a shape, aligning its axes with the trailing axes of that shape."""

    def __init__(self, target: Sequence[int | bounded], /) -> None: ...
    @property
    def target(self) -> tuple[int | bounded, ...]:
        """The shape this broadcasts to."""

class Constant(ProgramOp):
    """Supply a tensor the program holds, rather than one given at call time."""

    def __init__(self, value: ArrayLike, /) -> None: ...
    @property
    def value(self) -> np.ndarray:
        """The tensor this supplies, as a read-only array over the op's own buffer."""

class ShotLoop(ProgramOp):
    """Run each of several circuits for a number of shots."""

    def __init__(self, circuits: Sequence[QuantumCircuit], shots: int, /) -> None: ...
    @property
    def shots(self) -> int:
        """How many shots each circuit runs for."""

    @property
    def num_circuits(self) -> int:
        """How many circuits this runs, which is how many operands it takes."""

    def circuit(self, index: int, /) -> QuantumCircuit:
        """Return the circuit at ``index``."""

    def circuits(self) -> list[QuantumCircuit]:
        """Return every circuit this runs, in the order it takes its operands."""

class BindParameters(ProgramOp):
    """Evaluate each of several parameter expressions over a batch of values."""

    def __init__(
        self, expressions: Sequence[ParameterExpression], parameters: Sequence[Parameter], /
    ) -> None: ...
    @property
    def expressions(self) -> list[ParameterExpression]:
        """The expressions this evaluates, in the order it produces their values."""

    @property
    def parameters(self) -> list[Parameter]:
        """The parameters the values are for, in the order the operand holds them."""

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import ArrayLike

    from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
