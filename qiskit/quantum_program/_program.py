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

"""The Python-facing wrapper around a materialized quantum-program graph.

The graph itself (``qiskit._accelerate.quantum_program.QuantumProgram``) is a private PyO3 type,
built and only ever returned by :func:`.build`; this module wraps it together with the
:class:`.DataTree` of :class:`.Tracer` output ports that :func:`.build` was called with, so that
:meth:`QuantumProgram.resolve` can honour the round-trip contract: the structure you build with
is the structure you get back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["QuantumProgram"]


class QuantumProgram:
    """A materialized, type-resolvable quantum-program data-flow graph.

    Construct via :func:`.build`; the ``__init__`` signature is considered private. The
    underlying PyO3 graph is reachable (for introspection/testing) as :attr:`_graph`.
    """

    __slots__ = ("_graph", "_tree")

    def __init__(self, graph: _RustQuantumProgram, tree: DataTree) -> None:
        self._graph = graph
        self._tree = tree

    def resolve(self) -> Any:
        """Type-resolve the whole graph, returning the outputs in the shape :func:`.build` was
        called with (a bare :class:`.TensorSpec`, or the same list/tuple/dict/namedtuple nesting
        and structured-tracer shapes that were passed to :func:`.build` as ``outputs``)."""
        resolved = self._graph.resolve()
        return self._tree.unflatten(resolved.values()).to_python()

    def input_keys(self) -> list[str]:
        """The declared input keys."""
        return self._graph.input_keys()

    def output_keys(self) -> list[str]:
        """The dotted output keys backing :meth:`resolve`'s flattened leaves."""
        return self._graph.output_keys()

    def _node_labels(self) -> list[str]:
        """The labels of every materialized node, for introspection/testing."""
        return self._graph._node_labels()  # pylint: disable=protected-access

    def __repr__(self) -> str:
        return f"QuantumProgram(inputs={self.input_keys()}, outputs={self.output_keys()})"


if TYPE_CHECKING:
    from qiskit._accelerate.quantum_program import QuantumProgram as _RustQuantumProgram

    from ._datatree import DataTree
