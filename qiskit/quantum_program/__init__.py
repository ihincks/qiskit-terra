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

r"""
=========================================================
Quantum program construction (:mod:`qiskit.quantum_program`)
=========================================================

.. currentmodule:: qiskit.quantum_program

A :class:`QuantumProgram` is a data-flow graph of tensor operations. It is built with a
*tracer* front-end: operator calls on :class:`Tracer` objects assemble a pure-Python
expression tree, and no graph exists during construction. Each operation is type-checked
(dtype and shape) immediately, so mistakes are reported at the offending Python line, and
the real graph is materialized only when :func:`build` is called on the desired outputs::

    from qiskit.quantum_program import qp_input, build, f64

    x = qp_input("x", f64[3])
    y = (x * x).mean(axis=0)   # y.spec == TensorSpec(f64, []) is available now

    prog = build(outputs={"z": y})   # QuantumProgram materialized here
    prog.resolve()                   # {"z": TensorSpec(f64, [])}

Shared subexpressions collapse to a single node (fan-out) and tracers not reachable from
the requested outputs are dropped (dead-code elimination), both as a consequence of the
identity-memoized graph walk performed by :func:`build`.

A :class:`Tracer` doubles as a *port*: some ops (e.g. :func:`shot_loop`) have more than one
output, so they return a single, structured tracer -- indexable with ``[]``, ``len()``, and
``iter()`` like the ``list``/``dict`` it represents -- rather than a node-specific container
type. This is described by :class:`DataTree`, a leaf-or-branch structure that also underlies
``outputs`` itself, so :func:`build`/:meth:`~QuantumProgram.resolve` honour a round-trip
contract: the structure you build with is the structure you get back::

    theta = qp_input("theta", f64[1])
    result = shot_loop([circuit], shots=4000, params=[theta])  # one structured Tracer
    result[0]["meas"]                                          # index straight into it

    prog = build(result)            # no wrapping dict needed
    prog.resolve()                  # [{"meas": TensorSpec(bit, [4000, 2])}] -- same shape

Call :meth:`Tracer.draw` or :meth:`QuantumProgram.draw` to render the expression tree, either
as text (the default, with shared nodes tagged and expanded only once) or, with
``output="graphviz"``, as a :class:`PIL.Image.Image` graph layout -- one box per node, with
edges (wires) labelled by argument name, dtype/shape, and, for a port into a multi-output node,
its source path::

    print(prog.draw())
    # meas: shot_loop(shots=4000, circuits=1)[0]["meas"] → bit[4000, 2]
    #    └─ params[0]: input('theta') → f64[1]

    prog.draw(output="graphviz")   # PIL.Image.Image; requires Graphviz and Pillow

Every operator/method on :class:`Tracer` also has a numpy-style standalone-function
counterpart, so ``mean(x, axis=0)`` and ``x.mean(axis=0)`` (likewise ``add(x, y)``/``x + y``,
etc.) are interchangeable.

Functions
=========

.. autosummary::
   :toctree: ../stubs/

   qp_input
   constant
   shot_loop
   build
   add
   subtract
   multiply
   divide
   remainder
   power
   bitwise_and
   bitwise_or
   bitwise_xor
   bitwise_not
   mean
   var
   std
   parity

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   QuantumProgram
   Tracer
   TensorSpec
   DataTree
"""

from qiskit._accelerate.quantum_program import TensorSpec

from ._datatree import DataTree
from ._program import QuantumProgram
from ._tracer import (
    Tracer,
    add,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    build,
    constant,
    divide,
    mean,
    multiply,
    parity,
    power,
    qp_input,
    remainder,
    shot_loop,
    std,
    subtract,
    var,
)
from .dtypes import bit, c64, c128, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

__all__ = [
    "DataTree",
    "QuantumProgram",
    "TensorSpec",
    "Tracer",
    "add",
    "bit",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "build",
    "c64",
    "c128",
    "constant",
    "divide",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "mean",
    "multiply",
    "parity",
    "power",
    "qp_input",
    "remainder",
    "shot_loop",
    "std",
    "subtract",
    "u8",
    "u16",
    "u32",
    "u64",
    "var",
]
