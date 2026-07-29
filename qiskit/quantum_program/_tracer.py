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

"""A tracer front-end for building a :class:`.QuantumProgram`.

Operator calls on :class:`Tracer` objects build a pure-Python expression tree; **no**
:class:`.QuantumProgram` graph exists during construction. Each tracer eagerly computes its
output type by delegating to the Rust type-inference engine (``infer_op`` / ``infer_shot_loop``),
so dtype/shape mistakes are reported at the offending Python line.

A :class:`Tracer` doubles as a *port*: a shared, identity-keyed graph node (``_Node``) plus a
path into that node's output :class:`.DataTree`. This is what lets a single-output op (e.g.
``x + y``) and a multi-output op (e.g. :func:`shot_loop`) share one type -- indexing into a
structured tracer (``x[0]["meas"]``) is pure path navigation, not a graph edge, and only
resolves to a concrete node/edge when :func:`build` walks the tree.

The real graph is materialized only when :func:`build` walks the expression tree reachable
from the requested outputs. Sharing (common-subexpression / fan-out) falls out of an
``id``-keyed memo table over nodes (not tracers, since indexing the same node twice yields two
distinct tracer objects at the same port), and dead code (nodes not reachable from the outputs)
is simply never visited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from qiskit._accelerate.quantum_program import (
    QuantumProgram as _RustQuantumProgram,
    infer_op,
    infer_shot_loop,
    spec_of_array,
)

from ._datatree import DataTree
from ._program import QuantumProgram

__all__ = [
    "Tracer",
    "add",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "build",
    "constant",
    "divide",
    "mean",
    "multiply",
    "parity",
    "power",
    "qp_input",
    "remainder",
    "shot_loop",
    "std",
    "subtract",
    "var",
]

# Op names whose node has two inputs, wired to the "x" and "y" ports respectively.
_BINARY: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "remainder",
        "power",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    }
)

# Op names whose node has a single input, wired to the root port.
_UNARY: frozenset[str] = frozenset({"bitwise_not", "mean", "variance", "std", "parity"})


class _Node:
    """A graph node: an op, its operand tracers, and any op-specific attributes.

    Shared and identity-keyed -- multiple :class:`Tracer` ports (e.g. ``x[0]`` and ``x[1]`` of a
    structured tracer ``x``) may reference the same node.
    """

    __slots__ = ("args", "attrs", "op")

    def __init__(self, op: str, args: tuple[Tracer, ...], attrs: dict[str, Any]) -> None:
        self.op = op
        self.args = args
        self.attrs = attrs


class Tracer:
    """A port into a deferred :class:`.QuantumProgram` expression tree: a node plus a path.

    A leaf tracer carries a single :class:`.TensorSpec`, available eagerly via :attr:`spec`
    (and the :attr:`dtype`/:attr:`shape` sugar), and supports the usual Python arithmetic and
    bitwise operators plus :meth:`mean`/:meth:`var`/:meth:`std`/:meth:`parity` reductions. A
    structured tracer (e.g. the result of :func:`shot_loop`) instead carries a :class:`.DataTree`
    of child specs, navigable with ``[]``, ``len()``, ``iter()``, and (for keyed structures)
    :meth:`keys`/:meth:`values`/:meth:`items` -- indexing always returns another :class:`Tracer`,
    at the same node but a deeper path, until a leaf is reached.

    Tracers are compared and hashed *by identity* so that :func:`build` can memoize the
    materialization walk without accidentally merging distinct (possibly effectful) operations;
    the walk itself memoizes on the underlying node, not the tracer, since two tracers can share
    a node while pointing at different ports of it.
    """

    __slots__ = ("_node", "_path", "_tree")

    # Defer to our reflected dunders when the other operand is a numpy array or scalar,
    # rather than letting numpy broadcast us into an object array.
    __array_ufunc__ = None
    __array_priority__ = 1_000_000

    def __init__(self, node: _Node, path: list[int | str], tree: DataTree) -> None:
        self._node = node
        self._path = path
        self._tree = tree

    # -- identity ---------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    # -- structure ----------------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """Whether this port carries a single :class:`.TensorSpec` (vs. a structure of them)."""
        return self._tree.is_leaf

    @property
    def structure(self) -> DataTree:
        """The :class:`.DataTree` of :class:`.TensorSpec`\\ s at this port."""
        return self._tree

    def __datatree__(self) -> DataTree:
        if self.is_leaf:
            return DataTree.leaf(self)
        children = [
            Tracer(self._node, [*self._path, index if key is None else key], child).__datatree__()
            for index, (key, child) in enumerate(self._tree.iter_children())
        ]
        return DataTree.branch(self._tree.kind, children, self._tree.keys)

    def __len__(self) -> int:
        return len(self._tree)

    def __getitem__(self, key: int | str) -> Tracer:
        if self.is_leaf:
            raise TypeError(f"cannot index a leaf Tracer (dtype={self.dtype}, shape={self.shape})")
        if isinstance(key, str):
            child = self._tree.get_by_key(key)
            if child is None:
                raise KeyError(key)
        else:
            if not 0 <= key < len(self._tree):
                raise IndexError(key)
            child = self._tree.get(key)
        return Tracer(self._node, [*self._path, key], child)

    def __iter__(self) -> Iterator[Any]:
        if self.is_leaf:
            raise TypeError(
                f"cannot iterate a leaf Tracer (dtype={self.dtype}, shape={self.shape})"
            )
        if self._tree.kind == "dict":
            return iter(self.keys())
        return (self[index] for index in range(len(self._tree)))

    def keys(self) -> list[Any]:
        """The keys of a dict-shaped structured tracer."""
        self._require_dict("keys")
        return list(self._tree.keys)

    def values(self) -> list[Tracer]:
        """The child tracers of a dict-shaped structured tracer, keyed order."""
        return [self[key] for key in self.keys()]

    def items(self) -> list[tuple[Any, Tracer]]:
        """``(key, child tracer)`` pairs of a dict-shaped structured tracer."""
        return list(zip(self.keys(), self.values()))

    def _require_dict(self, what: str) -> None:
        if self.is_leaf or self._tree.kind != "dict":
            raise TypeError(f"{what}() is only valid on a dict-shaped Tracer")

    # -- inferred type ----------------------------------------------------------------------

    @property
    def spec(self) -> TensorSpec:
        """The leaf :class:`.TensorSpec`. Raises :class:`TypeError` on a structured tracer."""
        if not self.is_leaf:
            raise TypeError(
                f"spec is undefined on a structured Tracer (kind={self._tree.kind!r}, "
                f"{len(self._tree)} children) -- index into it first"
            )
        return self._tree.value

    @property
    def dtype(self) -> str:
        """The inferred dtype name of this tensor (e.g. ``"f64"``)."""
        return self.spec.dtype

    @property
    def shape(self) -> Shape:
        """The inferred shape of this tensor, as a list of ints and/or named dims."""
        return self.spec.shape

    # -- arithmetic operators ---------------------------------------------------------------

    def __add__(self, other: Operand) -> Tracer:
        return _binary("add", self, other)

    def __radd__(self, other: Operand) -> Tracer:
        return _binary("add", other, self)

    def __sub__(self, other: Operand) -> Tracer:
        return _binary("subtract", self, other)

    def __rsub__(self, other: Operand) -> Tracer:
        return _binary("subtract", other, self)

    def __mul__(self, other: Operand) -> Tracer:
        return _binary("multiply", self, other)

    def __rmul__(self, other: Operand) -> Tracer:
        return _binary("multiply", other, self)

    def __truediv__(self, other: Operand) -> Tracer:
        return _binary("divide", self, other)

    def __rtruediv__(self, other: Operand) -> Tracer:
        return _binary("divide", other, self)

    def __mod__(self, other: Operand) -> Tracer:
        return _binary("remainder", self, other)

    def __rmod__(self, other: Operand) -> Tracer:
        return _binary("remainder", other, self)

    def __pow__(self, other: Operand, modulo: Any = None) -> Tracer:
        if modulo is not None:
            raise ValueError("3-argument pow() is not supported by QuantumProgram tracers")
        return _binary("power", self, other)

    def __rpow__(self, other: Operand, modulo: Any = None) -> Tracer:
        if modulo is not None:
            raise ValueError("3-argument pow() is not supported by QuantumProgram tracers")
        return _binary("power", other, self)

    # -- bitwise operators ------------------------------------------------------------------

    def __and__(self, other: Operand) -> Tracer:
        return _binary("bitwise_and", self, other)

    def __rand__(self, other: Operand) -> Tracer:
        return _binary("bitwise_and", other, self)

    def __or__(self, other: Operand) -> Tracer:
        return _binary("bitwise_or", self, other)

    def __ror__(self, other: Operand) -> Tracer:
        return _binary("bitwise_or", other, self)

    def __xor__(self, other: Operand) -> Tracer:
        return _binary("bitwise_xor", self, other)

    def __rxor__(self, other: Operand) -> Tracer:
        return _binary("bitwise_xor", other, self)

    def __invert__(self) -> Tracer:
        return _unary("bitwise_not", self)

    # -- reductions -------------------------------------------------------------------------

    def mean(self, axis: int) -> Tracer:
        """Mean along ``axis``, removing that axis."""
        return _reduce("mean", self, axis)

    def var(self, axis: int, ddof: float = 0.0) -> Tracer:
        """Variance along ``axis``, removing that axis."""
        return _reduce("variance", self, axis, ddof)

    def std(self, axis: int, ddof: float = 0.0) -> Tracer:
        """Standard deviation along ``axis``, removing that axis."""
        return _reduce("std", self, axis, ddof)

    def parity(self, axis: int) -> Tracer:
        """XOR-reduction (parity) of a ``bit`` tensor along ``axis``, removing that axis."""
        return _reduce("parity", self, axis)

    # -- debugging --------------------------------------------------------------------------

    def draw(self, output: Literal["text", "graphviz"] = "text") -> Any:
        """Render this tracer's expression tree.

        ``output="text"`` (the default) returns a multi-line string. ``output="graphviz"``
        instead lays out the underlying node graph as an image (:class:`PIL.Image.Image`) to
        be viewed rather than printed, and requires the optional Graphviz and Pillow
        dependencies.
        """
        from ._draw import draw_tracers

        return draw_tracers(self.__datatree__(), output)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Tracer(op={self._node.op!r}, dtype={self.dtype}, shape={self.shape})"
        structure = self._tree.map_leaves(lambda spec: f"{spec.dtype}{spec.shape}").to_python()
        return f"Tracer(op={self._node.op!r}, structure={structure!r})"


# ------------------------------------------------------------------------------------------
# Factories
# ------------------------------------------------------------------------------------------


def _as_tracer(value: Operand) -> Tracer:
    """Return ``value`` if it is already a :class:`Tracer`, else wrap it as a constant."""
    if isinstance(value, Tracer):
        return value
    return constant(value)


def _leaf(node: _Node, spec: TensorSpec) -> Tracer:
    """A leaf :class:`Tracer` at the root port of a freshly built ``node``."""
    return Tracer(node, [], DataTree.leaf(spec))


def _binary(op: str, lhs: Operand, rhs: Operand) -> Tracer:
    """Build a binary-op tracer, eagerly inferring its output spec."""
    lhs, rhs = _as_tracer(lhs), _as_tracer(rhs)
    (spec,) = infer_op(op, [lhs.spec, rhs.spec])
    return _leaf(_Node(op, (lhs, rhs), {}), spec)


def _unary(op: str, operand: Operand) -> Tracer:
    """Build a unary-op tracer (no extra attributes), eagerly inferring its output spec."""
    operand = _as_tracer(operand)
    (spec,) = infer_op(op, [operand.spec])
    return _leaf(_Node(op, (operand,), {}), spec)


def _reduce(op: str, operand: Operand, axis: int, ddof: float | None = None) -> Tracer:
    """Build a reduction-op tracer along ``axis`` (with optional ``ddof``)."""
    operand = _as_tracer(operand)
    if ddof is None:
        (spec,) = infer_op(op, [operand.spec], axis=axis)
        attrs: dict[str, Any] = {"axis": axis}
    else:
        (spec,) = infer_op(op, [operand.spec], axis=axis, ddof=ddof)
        attrs = {"axis": axis, "ddof": ddof}
    return _leaf(_Node(op, (operand,), attrs), spec)


# ------------------------------------------------------------------------------------------
# Standalone ops -- a numpy-style alternative to the Tracer operators/methods, so callers can
# write either ``x.mean(axis=0)`` or ``mean(x, axis=0)``.
# ------------------------------------------------------------------------------------------


def add(x: Operand, y: Operand) -> Tracer:
    """Elementwise addition. Equivalent to ``x + y``."""
    return _binary("add", x, y)


def subtract(x: Operand, y: Operand) -> Tracer:
    """Elementwise subtraction. Equivalent to ``x - y``."""
    return _binary("subtract", x, y)


def multiply(x: Operand, y: Operand) -> Tracer:
    """Elementwise multiplication. Equivalent to ``x * y``."""
    return _binary("multiply", x, y)


def divide(x: Operand, y: Operand) -> Tracer:
    """Elementwise division. Equivalent to ``x / y``."""
    return _binary("divide", x, y)


def remainder(x: Operand, y: Operand) -> Tracer:
    """Elementwise remainder. Equivalent to ``x % y``."""
    return _binary("remainder", x, y)


def power(x: Operand, y: Operand) -> Tracer:
    """Elementwise exponentiation. Equivalent to ``x ** y``."""
    return _binary("power", x, y)


def bitwise_and(x: Operand, y: Operand) -> Tracer:
    """Elementwise bitwise AND. Equivalent to ``x & y``."""
    return _binary("bitwise_and", x, y)


def bitwise_or(x: Operand, y: Operand) -> Tracer:
    """Elementwise bitwise OR. Equivalent to ``x | y``."""
    return _binary("bitwise_or", x, y)


def bitwise_xor(x: Operand, y: Operand) -> Tracer:
    """Elementwise bitwise XOR. Equivalent to ``x ^ y``."""
    return _binary("bitwise_xor", x, y)


def bitwise_not(x: Operand) -> Tracer:
    """Elementwise bitwise NOT. Equivalent to ``~x``."""
    return _unary("bitwise_not", x)


def mean(x: Operand, axis: int) -> Tracer:
    """Mean along ``axis``, removing that axis. Equivalent to ``x.mean(axis)``."""
    return _reduce("mean", x, axis)


def var(x: Operand, axis: int, ddof: float = 0.0) -> Tracer:
    """Variance along ``axis``, removing that axis. Equivalent to ``x.var(axis, ddof)``."""
    return _reduce("variance", x, axis, ddof)


def std(x: Operand, axis: int, ddof: float = 0.0) -> Tracer:
    """Standard deviation along ``axis``, removing that axis. Equivalent to ``x.std(axis, ddof)``."""
    return _reduce("std", x, axis, ddof)


def parity(x: Operand, axis: int) -> Tracer:
    """XOR-reduction (parity) along ``axis``, removing that axis. Equivalent to ``x.parity(axis)``."""
    return _reduce("parity", x, axis)


def qp_input(key: str, spec: TensorSpec) -> Tracer:
    """Declare a program input named ``key`` with the given :class:`.TensorSpec`.

    Returns a :class:`Tracer` carrying that spec; the same input may fan out to any number of
    downstream operations.
    """
    return _leaf(_Node("input", (), {"key": key, "spec": spec}), spec)


def constant(value: ArrayLike) -> Tracer:
    """Wrap a numpy array-like ``value`` as a constant :class:`Tracer`."""
    array = np.asarray(value)
    return _leaf(_Node("constant", (), {"value": array}), spec_of_array(array))


def shot_loop(
    circuits: Iterable[QuantumCircuit],
    shots: int,
    params: Sequence[Operand],
) -> Tracer:
    """Run each of ``circuits`` for ``shots`` shots, fed by per-circuit ``params``.

    ``params[i]`` supplies the parameter values for ``circuits[i]``, in that circuit's
    parameter order; it must be an ``f64`` tensor of shape ``[..., num_parameters_i]`` (any
    leading axes are an opaque batch prefix).

    Returns a single structured :class:`Tracer` shaped like a list with one entry per circuit,
    each entry a dict mapping that circuit's classical-register names to a ``bit`` tracer of
    shape ``[..., shots, register_len]`` holding the per-shot measurement outcomes. Index into
    it (``result[0]["meas"]``) to reach the leaf tracers; every such leaf shares this call's
    single, effectful ``shot_loop`` node.
    """
    circuits = list(circuits)
    param_tracers = [_as_tracer(p) for p in params]
    per_circuit = infer_shot_loop(circuits, shots, [p.spec for p in param_tracers])
    node = _Node("shot_loop", tuple(param_tracers), {"circuits": circuits, "shots": shots})
    return Tracer(node, [], DataTree.from_python(per_circuit))


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------


def build(outputs: Outputs) -> QuantumProgram:
    """Materialize a :class:`.QuantumProgram` from a tracer expression tree.

    ``outputs`` names the program's declared outputs: a single :class:`Tracer` (leaf or
    structured), or any nesting of ``list``/``tuple``/``dict``/namedtuple containing tracers.
    Only the nodes reachable from ``outputs`` are materialized: shared subexpressions collapse
    to a single node (fan-out) and unreferenced nodes are dropped (dead-code elimination), both
    as a natural consequence of the ``id``-memoized graph walk.

    The returned :class:`.QuantumProgram`'s :meth:`~.QuantumProgram.resolve` honours a
    round-trip contract: the structure you build with is the structure you get back.
    """
    tree = DataTree.from_python(outputs)
    paths = tree.paths()
    dotted_keys = tree.dotted_paths()
    leaves = tree.leaves()

    seen_at: dict[str, list[list[int | str]]] = {}
    for path, key, leaf in zip(paths, dotted_keys, leaves):
        if not isinstance(leaf, Tracer):
            raise TypeError(
                f"build() output at {key!r} is not a Tracer (got {type(leaf).__name__!r})"
            )
        seen_at.setdefault(key, []).append(path)
    duplicates = sorted(key for key, at in seen_at.items() if len(at) > 1)
    if duplicates:
        raise ValueError(f"build() outputs have colliding dotted keys: {duplicates!r}")

    program = _RustQuantumProgram()

    label_memo: dict[int, str] = {}  # id(_Node) -> its materialized label
    seen: dict[int, _Node] = {}  # id(_Node) -> node; pins lifetimes vs. id() recycling
    counters: dict[str, int] = {}  # op name -> next auto-label index
    input_labels: dict[str, tuple[str, TensorSpec]] = {}  # key -> (label, spec) for dedup

    def fresh_label(op: str) -> str:
        index = counters.get(op, 0)
        counters[op] = index + 1
        return f"{op}_{index}"

    def port_of(tracer: Tracer) -> Port:
        return (materialize(tracer._node), list(tracer._path))  # pylint: disable=protected-access

    def materialize(node: _Node) -> str:
        key = id(node)
        if key in label_memo:
            return label_memo[key]
        seen[key] = node
        op = node.op

        if op == "input":
            input_key, spec = node.attrs["key"], node.attrs["spec"]
            if input_key in input_labels:
                label, existing_spec = input_labels[input_key]
                if existing_spec != spec:
                    raise ValueError(
                        f"input key {input_key!r} declared with conflicting specs "
                        f"{existing_spec!r} and {spec!r}"
                    )
            else:
                label = program._add_input(input_key, spec)
                input_labels[input_key] = (label, spec)
        elif op == "constant":
            label = fresh_label("constant")
            program._add_constant(label, node.attrs["value"])
        elif op == "shot_loop":
            param_ports = [port_of(param) for param in node.args]
            label = fresh_label("shot_loop")
            program._add_shot_loop(label, node.attrs["circuits"], node.attrs["shots"])
            for index, (from_label, from_path) in enumerate(param_ports):
                program._add_edge(from_label, from_path, label, [index])
        elif op in _BINARY:
            lhs_port = port_of(node.args[0])
            rhs_port = port_of(node.args[1])
            label = fresh_label(op)
            program._add_op(label, op)
            program._add_edge(lhs_port[0], lhs_port[1], label, ["x"])
            program._add_edge(rhs_port[0], rhs_port[1], label, ["y"])
        elif op in _UNARY:
            operand_port = port_of(node.args[0])
            label = fresh_label(op)
            program._add_op(label, op, axis=node.attrs.get("axis"), ddof=node.attrs.get("ddof"))
            program._add_edge(operand_port[0], operand_port[1], label, [])
        else:
            raise ValueError(f"cannot materialize node with unknown op {op!r}")

        label_memo[key] = label
        return label

    for key, leaf in zip(dotted_keys, leaves):
        label, path = port_of(leaf)
        program._set_output(key, label, path)

    return QuantumProgram(program, tree)


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.typing import ArrayLike

    from qiskit.circuit import QuantumCircuit
    from qiskit._accelerate.quantum_program import TensorSpec

    # An inferred tensor shape: fixed dims (``int``) and/or named symbolic dims (``str``).
    Shape = list[int | str]
    # A tracer operand: another tracer, or any array-like coercible to a constant.
    Operand = Tracer | ArrayLike
    # The ``outputs`` argument accepted by :func:`build`.
    Outputs = Tracer | dict[Any, Any] | list[Any] | tuple[Any, ...]
    # A materialized output port: a node label plus a path into that node's output tree.
    Port = tuple[str, list[int | str]]
