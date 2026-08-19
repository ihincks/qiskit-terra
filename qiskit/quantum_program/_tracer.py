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

"""Tracer and functions that return it"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from qiskit._accelerate.quantum_program import (
    DataTree,
    OpNodeType,
    ProgramFunction,
)

__all__ = [
    "Tracer",
    "add",
    "bind_parameters",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_to",
    "build",
    "cast",
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


class _Node:
    """One node of a program being written: the values it consumes and the types it produces."""

    __slots__ = ("operands", "types")

    def __init__(self, operands: tuple[Tracer, ...], types: Types) -> None:
        self.operands = operands
        self.types = types


class _Input(_Node):
    """A declared program input, named for the keyword argument that will supply it."""

    __slots__ = ("name",)

    def __init__(self, name: str, type_: TensorType) -> None:
        super().__init__((), type_)
        self.name = name

    def __repr__(self) -> str:
        return f"input {self.name!r}"


class _Operation(_Node):
    """An operation applied to operands, typed as it is created."""

    __slots__ = ("node_type",)

    def __init__(self, node_type: OpNodeType, operands: tuple[Tracer, ...]) -> None:
        types = node_type.output_types([operand.type for operand in operands])
        # One result is carried as its type, which is what indexing a structure gives for a value.
        super().__init__(operands, types.leaf if types.is_leaf else types)
        self.node_type = node_type

    def __repr__(self) -> str:
        return self.node_type.full_name


class Tracer:
    """One value of a quantum program being constructed, or a structure of them.

    A quantum program is not constructed directly. Instead, it is constructed incrementally, one
    operation at a time, and a tracer object represents that incremental progress: it stands for a
    tensor value the program will produce, together with the operations that produce it, and it can
    be an operand of further operations. Once every operation has been added, :func:`build` turns
    the tracers into a program, while also allowing them to be named and structured.

    Tracers are not constructed directly: every function and method that produces a value returns
    one. :func:`qp_input`, :func:`constant` and :func:`shot_loop` start an expression, and each
    operation extends one. A tracer also has an operator or a method for each operation, so
    ``x + y`` and :func:`add(x, y) <add>` are equivalent::

        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_program import build, f64, qp_input, shot_loop

        circuit = QuantumCircuit(2)
        circuit.measure_all()

        outcomes = shot_loop([circuit], shots=1024)  # a tracer over all of the loop's results
        bits = outcomes[0]["meas"]                   # one value, at a position in that loop
        weights = qp_input("weights", f64[2])        # one value, supplied when the program runs
        signal = (bits * weights).mean(axis=0)       # as mean(multiply(bits, weights), 0)
        signal.type                                  # TensorType(F64[2])
        program = build({"signal": signal - 0.5})    # 0.5 is carried as a constant

    An operand that is not a tracer, such as the ``0.5`` above, becomes a :func:`constant`. An
    operation promotes the dtypes and broadcasts the shapes of its operands itself, so those
    conversions are implicit in the operation that performs them rather than nodes of their own.

    An operation producing several values, such as :func:`shot_loop`, returns one tracer over all of
    them, arranged as that operation arranges its results. Such a tracer is navigated with ``[]``,
    :func:`len`, ``in``, iteration and, where its values are named, :meth:`keys`, :meth:`values` and
    :meth:`items`. Indexing reaches one value, which is where :attr:`type` is defined, and each
    value reached is a position in the one operation.

    Tracer equality is defined by identity, which allows :func:`build` to collapse a value used
    twice into a single node.
    """

    __slots__ = ("_node", "_slot", "_types")

    # NumPy defers to the reflected operators here rather than broadcasting a tracer into an array
    # of objects.
    __array_ufunc__ = None

    def __init__(self, node: _Node, slot: int, types: Types) -> None:
        self._node = node
        self._slot = slot
        self._types = types

    @property
    def is_leaf(self) -> bool:
        """Whether this is one value, as opposed to a structure of them."""
        return not isinstance(self._types, DataTree)

    @property
    def type(self) -> TensorType:
        """The type of this value, inferred where it was written."""
        if not self.is_leaf:
            raise TypeError("a structure of values has no type of its own; index into it first")
        return self._types

    @property
    def dtype(self) -> DType:
        """The dtype of this value."""
        return self.type.dtype

    @property
    def shape(self) -> tuple[int | bounded, ...]:
        """The shape of this value."""
        return self.type.shape

    def __bool__(self) -> bool:
        # Without this, `__len__` answers a truth test, which is not a question about the structure.
        return True

    def __len__(self) -> int:
        return len(self._structure("no length"))

    def __iter__(self) -> Iterator[Tracer]:
        count = len(self._structure("nothing to iterate"))
        return (self[position] for position in range(count))

    def __getitem__(self, key: int | str) -> Tracer:
        types = self._structure("nothing to index")
        children = list(types)
        if isinstance(key, str):
            if not types.is_mapping:
                raise TypeError("a structure of unnamed values is indexed by position")
            names = types.keys()
            if key not in names:
                raise KeyError(key)
            position = names.index(key)
        elif hasattr(key, "__index__"):
            position = operator.index(key)
            position = position + len(children) if position < 0 else position
            if not 0 <= position < len(children):
                raise IndexError(f"position {key} addresses nothing among {len(children)} values")
        else:
            raise TypeError(
                "a structure of values is indexed by name or by position, not by "
                f"{type(key).__name__}"
            )
        slot = self._slot + sum(_leaf_count(child) for child in children[:position])
        return Tracer(self._node, slot, children[position])

    def __contains__(self, key: object) -> bool:
        try:
            self[key]
        except (IndexError, KeyError, TypeError):
            return False
        return True

    def keys(self) -> list[str]:
        """The name of each value one level down, in order.

        Returns:
            The names, in the order the values sit in.

        Raises:
            TypeError: If this is one value, or a structure naming none of its values.
        """
        types = self._structure("no mapping form")
        if not types.is_mapping:
            raise TypeError("a structure of unnamed values has no mapping form")
        return types.keys()

    def values(self) -> list[Tracer]:
        """Each value one level down, in order.

        Returns:
            A tracer for each, as indexing gives it.

        Raises:
            TypeError: If this is one value, or a structure naming none of its values.
        """
        return [self[name] for name in self.keys()]

    def items(self) -> list[tuple[str, Tracer]]:
        """Each value one level down with its name, in order.

        Returns:
            A pair per value, of its name and the tracer :meth:`values` gives for it.

        Raises:
            TypeError: If this is one value, or a structure naming none of its values.
        """
        return list(zip(self.keys(), self.values()))

    def __datatree__(self) -> DataTree:
        if self.is_leaf:
            return DataTree.leaf_of(self)
        if self._types.is_mapping:
            return DataTree(dict(self.items()))
        return DataTree(list(self))

    def _structure(self, missing: str) -> DataTree:
        """The types of the values one level down, or the error saying this is one value."""
        if self.is_leaf:
            raise TypeError(f"a single value of type {self._types} has {missing}")
        return self._types

    def __add__(self, other: Operand) -> Tracer:
        return add(self, other)

    def __radd__(self, other: Operand) -> Tracer:
        return add(other, self)

    def __sub__(self, other: Operand) -> Tracer:
        return subtract(self, other)

    def __rsub__(self, other: Operand) -> Tracer:
        return subtract(other, self)

    def __mul__(self, other: Operand) -> Tracer:
        return multiply(self, other)

    def __rmul__(self, other: Operand) -> Tracer:
        return multiply(other, self)

    def __truediv__(self, other: Operand) -> Tracer:
        return divide(self, other)

    def __rtruediv__(self, other: Operand) -> Tracer:
        return divide(other, self)

    def __mod__(self, other: Operand) -> Tracer:
        return remainder(self, other)

    def __rmod__(self, other: Operand) -> Tracer:
        return remainder(other, self)

    def __pow__(self, other: Operand) -> Tracer:
        return power(self, other)

    def __rpow__(self, other: Operand) -> Tracer:
        return power(other, self)

    def __and__(self, other: Operand) -> Tracer:
        return bitwise_and(self, other)

    def __rand__(self, other: Operand) -> Tracer:
        return bitwise_and(other, self)

    def __or__(self, other: Operand) -> Tracer:
        return bitwise_or(self, other)

    def __ror__(self, other: Operand) -> Tracer:
        return bitwise_or(other, self)

    def __xor__(self, other: Operand) -> Tracer:
        return bitwise_xor(self, other)

    def __rxor__(self, other: Operand) -> Tracer:
        return bitwise_xor(other, self)

    def __invert__(self) -> Tracer:
        return bitwise_not(self)

    def mean(self, axis: int) -> Tracer:
        """Average along ``axis``, removing it. Equivalent to :func:`mean(self, axis) <mean>`.

        Args:
            axis: The axis to average along.

        Returns:
            The average, with ``axis`` removed.
        """
        return mean(self, axis)

    def var(self, axis: int, ddof: float = 0.0) -> Tracer:
        """Variance along ``axis``, removing it. Equivalent to :func:`var(self, axis, ddof) <var>`.

        Args:
            axis: The axis to take the variance along.
            ddof: The delta degrees of freedom, subtracted from the divisor.

        Returns:
            The variance, with ``axis`` removed.
        """
        return var(self, axis, ddof)

    def std(self, axis: int, ddof: float = 0.0) -> Tracer:
        """Deviation along ``axis``, removing it. Equivalent to :func:`std(self, axis, ddof) <std>`.

        Args:
            axis: The axis to take the standard deviation along.
            ddof: The delta degrees of freedom, subtracted from the divisor.

        Returns:
            The standard deviation, with ``axis`` removed.
        """
        return std(self, axis, ddof)

    def parity(self, axis: int) -> Tracer:
        """Parity along ``axis``, removing it. Equivalent to :func:`parity(self, axis) <parity>`.

        Args:
            axis: The axis to XOR-reduce along.

        Returns:
            The parity of the bits along ``axis``, with that axis removed.
        """
        return parity(self, axis)

    def __repr__(self) -> str:
        return f"Tracer({self._node!r}, {self._types})"


def qp_input(name: str, type_: TensorType) -> Tracer:
    """Declare an input of type ``type_``, supplied by the keyword argument ``name``.

    Args:
        name: The keyword argument that will supply this input.
        type_: The dtype and shape the program demands of it.

    Returns:
        A tracer for the input. Using it in several expressions still declares one input.
    """
    return _tracer(_Input(name, type_))


def constant(value: ArrayLike) -> Tracer:
    """A value the program holds, rather than one supplied at call time.

    Args:
        value: The value to hold, read with :func:`numpy.asarray`.

    Returns:
        A tracer whose dtype and shape are those of the resulting array.
    """
    return _operation(OpNodeType.constant(value))


def add(x: Operand, y: Operand) -> Tracer:
    """Add ``x`` and ``y`` elementwise. Equivalent to ``x + y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise sum.
    """
    return _operation(OpNodeType.add(), x, y)


def subtract(x: Operand, y: Operand) -> Tracer:
    """Subtract ``y`` from ``x`` elementwise. Equivalent to ``x - y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise difference.
    """
    return _operation(OpNodeType.subtract(), x, y)


def multiply(x: Operand, y: Operand) -> Tracer:
    """Multiply ``x`` and ``y`` elementwise. Equivalent to ``x * y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise product.
    """
    return _operation(OpNodeType.multiply(), x, y)


def divide(x: Operand, y: Operand) -> Tracer:
    """Divide ``x`` by ``y`` elementwise. Equivalent to ``x / y``.

    A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.

    Args:
        x: The dividend.
        y: The divisor.

    Returns:
        The elementwise quotient.
    """
    return _operation(OpNodeType.divide(), x, y)


def remainder(x: Operand, y: Operand) -> Tracer:
    """Remainder of ``x`` divided by ``y``, elementwise. Equivalent to ``x % y``.

    A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.

    Args:
        x: The dividend.
        y: The divisor.

    Returns:
        The elementwise remainder.
    """
    return _operation(OpNodeType.remainder(), x, y)


def power(x: Operand, y: Operand) -> Tracer:
    """Raise ``x`` to the power ``y`` elementwise. Equivalent to ``x ** y``.

    Args:
        x: The base.
        y: The exponent.

    Returns:
        The elementwise power.
    """
    return _operation(OpNodeType.power(), x, y)


def bitwise_and(x: Operand, y: Operand) -> Tracer:
    """Bitwise AND of ``x`` and ``y``, elementwise. Equivalent to ``x & y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise AND.
    """
    return _operation(OpNodeType.bitwise_and(), x, y)


def bitwise_or(x: Operand, y: Operand) -> Tracer:
    """Bitwise OR of ``x`` and ``y``, elementwise. Equivalent to ``x | y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise OR.
    """
    return _operation(OpNodeType.bitwise_or(), x, y)


def bitwise_xor(x: Operand, y: Operand) -> Tracer:
    """Bitwise XOR of ``x`` and ``y``, elementwise. Equivalent to ``x ^ y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise exclusive OR.
    """
    return _operation(OpNodeType.bitwise_xor(), x, y)


def bitwise_not(x: Operand) -> Tracer:
    """Bitwise NOT of ``x``, elementwise. Equivalent to ``~x``.

    Args:
        x: The bits to invert.

    Returns:
        The elementwise NOT.
    """
    return _operation(OpNodeType.bitwise_not(), x)


def mean(x: Operand, axis: int) -> Tracer:
    """Average ``x`` along ``axis``, removing that axis.

    Args:
        x: The value to reduce.
        axis: The axis to average along.

    Returns:
        The average, with ``axis`` removed. Averaging bits gives a float.
    """
    return _operation(OpNodeType.mean(axis), x)


def var(x: Operand, axis: int, ddof: float = 0.0) -> Tracer:
    """Variance of ``x`` along ``axis``, removing that axis.

    The sum of squared deviations is divided by ``n - ddof``, where ``n`` is the length of ``axis``.

    Args:
        x: The value to reduce.
        axis: The axis to take the variance along.
        ddof: The delta degrees of freedom, subtracted from the divisor.

    Returns:
        The variance, with ``axis`` removed.
    """
    return _operation(OpNodeType.variance(axis, ddof), x)


def std(x: Operand, axis: int, ddof: float = 0.0) -> Tracer:
    """Standard deviation of ``x`` along ``axis``, removing that axis.

    This is the square root of :func:`var`, and ``ddof`` means the same thing.

    Args:
        x: The value to reduce.
        axis: The axis to take the standard deviation along.
        ddof: The delta degrees of freedom, subtracted from the divisor.

    Returns:
        The standard deviation, with ``axis`` removed.
    """
    return _operation(OpNodeType.std(axis, ddof), x)


def parity(x: Operand, axis: int) -> Tracer:
    """XOR-reduce the bits of ``x`` along ``axis``, removing that axis.

    Args:
        x: The bits to reduce.
        axis: The axis to XOR-reduce along.

    Returns:
        The parity of the bits along ``axis``, with that axis removed.
    """
    return _operation(OpNodeType.parity(axis), x)


def cast(x: Operand, dtype: DType) -> Tracer:
    """Reinterpret ``x`` as ``dtype``, keeping its shape.

    Args:
        x: The value to cast.
        dtype: The dtype to cast to. A complex value cannot be cast to a real dtype.

    Returns:
        The value in ``dtype``.
    """
    return _operation(OpNodeType.cast(dtype), x)


def broadcast_to(x: Operand, shape: Sequence[int | bounded]) -> Tracer:
    """Broadcast ``x`` to ``shape``, aligning its axes with the trailing axes of ``shape``.

    Args:
        x: The value to broadcast.
        shape: The shape to reach. Each axis of ``x`` must either match the axis it aligns with
            or have size one.

    Returns:
        A value of shape ``shape``.
    """
    return _operation(OpNodeType.broadcast_to(shape), x)


def shot_loop(
    circuits: Iterable[QuantumCircuit],
    shots: int,
    parameter_values: Sequence[Operand | None] | None = None,
) -> Tracer:
    """Run each of ``circuits`` for ``shots`` shots, over the values given for its parameters.

    ``parameter_values[i]`` has the values for ``circuits[i]`` in its trailing axis, one per
    parameter of that circuit in the order :attr:`.QuantumCircuit.parameters` gives them. Its
    leading axes are a batch of parameterizations, and are carried onto that circuit's outcomes.

    A circuit taking no parameters takes no values, written as ``None`` in its place, or by leaving
    ``parameter_values`` out when no circuit takes any. Its outcomes then have no batch.

    Args:
        circuits: The circuits to run, each copied as it is wired in.
        shots: How many shots to run each circuit for.
        parameter_values: One entry per circuit, holding that circuit's parameter values, or
            ``None`` where a circuit takes none.

    Returns:
        The outcomes of every classical register, as a sequence over the circuits of a mapping from
        register name. Each is ``bit[..., shots, register width]``, over the batch of that circuit's
        values. Index into it to reach one register's outcomes.

    Raises:
        TypeError: If an entry of ``circuits`` is not a circuit.
        ValueError: If there is not one entry of ``parameter_values`` per circuit, if an entry does
            not have that circuit's parameter values, or if a register's name cannot name a value.
    """
    circuits = list(circuits)
    if parameter_values is None:
        parameter_values = [None] * len(circuits)
    else:
        parameter_values = list(parameter_values)
    if len(parameter_values) != len(circuits):
        raise ValueError(
            f"one set of parameter values is needed per circuit: {len(circuits)} circuits, "
            f"{len(parameter_values)} given"
        )
    if any(entry is None for entry in parameter_values):
        # One empty set of values, shared by every circuit taking no parameters.
        empty = constant([])
        parameter_values = [empty if entry is None else entry for entry in parameter_values]
    return _operation(OpNodeType.shot_loop(circuits, shots), *parameter_values)


def bind_parameters(
    expressions: Sequence[ParameterExpression],
    parameter_values: Operand,
    parameters: Sequence[Parameter],
) -> Tracer:
    """Evaluate each of ``expressions`` over a batch of values for ``parameters``.

    ``parameter_values`` has one value per entry of ``parameters`` in its trailing axis, in that
    order. Its leading axes are a batch of parameterizations, and are carried onto the result.

    Args:
        expressions: The expressions to evaluate.
        parameter_values: The values, as a floating-point value of shape ``[..., len(parameters)]``.
        parameters: The parameters the values are for. Every parameter an expression references must
            appear here, and surplus ones are ignored, so a whole circuit's parameters can be passed
            as they are.

    Returns:
        A value of shape ``[..., len(expressions)]``, holding each expression's value in the
        trailing axis. Expressions evaluate in double precision, so it is ``f64``.

    Raises:
        ValueError: If an expression references a parameter ``parameters`` does not name, or if the
            values are not a floating-point value of the right trailing size.
    """
    node_type = OpNodeType.bind_parameters(list(expressions), list(parameters))
    return _operation(node_type, parameter_values)


def build(outputs: Any) -> QuantumProgram:
    """Build the program producing ``outputs``.

    The input can be a tracer, or any nesting of sequences and mappings of them: the structure
    provided becomes the program's output structure.

    For example::

        x = qp_input("x", f64[3])
        program = build({"mean": x.mean(0), "shifted": x - 1.0})

        program(x=[1.0, 2.0, 3.0])
        # DataTree([mean: array(2.), shifted: array([0., 1., 2.])])

    Args:
        outputs: One tracer, or any nesting of sequences and mappings of them.

    Returns:
        The program, declaring the inputs and outputs the walk found.

    Raises:
        TypeError: If an output is not a tracer.
        ValueError: If two inputs are declared under one name, or a name cannot address a value.
    """
    outputs = DataTree(outputs)
    function = ProgramFunction()
    # Keyed by `id`, which identifies a node only while it is alive. Every node reachable from
    # `outputs` is, since each one holds the tracers it consumes.
    values: dict[int, list[Value]] = {}
    inputs: dict[str, _Input] = {}

    def materialize(tracer: Tracer) -> Value:
        """The value ``tracer`` stands for, adding its node and everything below it to the function."""
        stack = [tracer._node]
        while stack:
            next_node = stack[-1]
            if id(next_node) in values:
                stack.pop()
                continue
            unbuilt = [
                operand._node for operand in next_node.operands if id(operand._node) not in values
            ]
            if unbuilt:
                # Reversed, so that the stack pops the operands left to right and a program
                # declares its inputs in the order they were written.
                stack.extend(reversed(unbuilt))
                continue
            stack.pop()
            if isinstance(next_node, _Input):
                if inputs.setdefault(next_node.name, next_node) is not next_node:
                    raise ValueError(f"input {next_node.name!r} is declared twice")
                values[id(next_node)] = [function.add_parameter(next_node.types)]
            else:
                operands = [
                    values[id(operand._node)][operand._slot] for operand in next_node.operands
                ]
                values[id(next_node)] = function.add_node(next_node.node_type, operands)
        return values[id(tracer._node)][tracer._slot]

    for path, leaf in _leaves(outputs):
        where = f"output {path!r}" if path else "the output"
        if not isinstance(leaf, Tracer):
            raise TypeError(f"{where} is a {type(leaf).__name__}, not a Tracer")
        if not leaf.is_leaf:
            raise TypeError(f"{where} is a structure of values, not one value")
        function.add_result(materialize(leaf))
    return function.seal(DataTree(dict.fromkeys(inputs)), outputs)


def _tracer(node: _Node) -> Tracer:
    """A tracer over everything ``node`` produces."""
    return Tracer(node, 0, node.types)


def _operation(node_type: OpNodeType, *operands: Operand) -> Tracer:
    """Apply ``node_type`` to ``operands``, wrapping any that are not tracers as constants."""
    return _tracer(_Operation(node_type, tuple(_as_tracer(operand) for operand in operands)))


def _as_tracer(value: Operand) -> Tracer:
    """``value`` itself if it is a tracer, and a constant holding it otherwise."""
    return value if isinstance(value, Tracer) else constant(value)


def _leaf_count(types: Types) -> int:
    """How many values ``types`` covers: one, or every value in a structure of them."""
    if not isinstance(types, DataTree):
        return 1
    return sum(_leaf_count(child) for child in types)


def _leaves(tree: DataTree, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every leaf of ``tree`` in order, each with the dotted path that addresses it."""
    if tree.is_leaf:
        yield path, tree.leaf
        return
    children = tree.items() if tree.is_mapping else enumerate(tree)
    for key, child in children:
        below = f"{path}.{key}" if path else str(key)
        if isinstance(child, DataTree):
            yield from _leaves(child, below)
        else:
            yield below, child


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import TypeAlias

    from numpy.typing import ArrayLike

    from qiskit._accelerate.quantum_program import DType, QuantumProgram, TensorType, Value, bounded
    from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit

    # A tracer, or anything `numpy.asarray` reads, which becomes a constant.
    Operand: TypeAlias = Tracer | ArrayLike
    # The types at one position in a node's results: one value's type, or a structure of them.
    Types: TypeAlias = TensorType | DataTree
