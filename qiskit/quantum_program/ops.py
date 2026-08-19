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

"""
=================================================
Program ops (:mod:`qiskit.quantum_program.ops`)
=================================================

.. currentmodule:: qiskit.quantum_program.ops

An op is one of the operations an instruction of a :class:`~qiskit.quantum_program.QuantumProgram`
can perform. It holds whatever the operation needs beyond its operands, such as the axis a reduction
folds along or the circuits a shot loop runs, and it reports the types it produces from the types it
is given.

Each operation has a class here, and one of those is what an instruction of a built program hands
back. The class both builds an operation and reads the payload it was built with::

    from qiskit.quantum_program import ops

    op = ops.Mean(0)
    op.full_name       # 'qiskit.mean'
    op.axis            # 0

The functions in :mod:`qiskit.quantum_program`, such as :func:`~qiskit.quantum_program.mean`, are
what writes a program, and they apply these operations for you. Reach for a class here to read an
operation back, or to apply one directly.

An op defined outside Qiskit reads back as :class:`ProgramOp` itself, which reports its name and a
summary of its payload and nothing else.

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   ProgramOp
   Add
   Subtract
   Multiply
   Divide
   Remainder
   Power
   BitwiseAnd
   BitwiseOr
   BitwiseXor
   BitwiseNot
   Parity
   Mean
   Variance
   Std
   Cast
   BroadcastTo
   Constant
"""

from qiskit._accelerate.quantum_program import (
    Add,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    BroadcastTo,
    Cast,
    Constant,
    Divide,
    Mean,
    Multiply,
    Parity,
    Power,
    ProgramOp,
    Remainder,
    Std,
    Subtract,
    Variance,
)

__all__ = [
    "Add",
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
    "Std",
    "Subtract",
    "Variance",
]
