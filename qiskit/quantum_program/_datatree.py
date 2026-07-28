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

"""A tree of leaf values, addressable by index and/or string key.

:class:`DataTree` mirrors the Rust ``DataTree`` (``crates/providers/src/data_tree.rs``) used
internally by :class:`.QuantumProgram` nodes with structured outputs (e.g. ``shot_loop``): a
branch holds an ordered sequence of children, any of which may additionally carry a string
key. This is *not* a binding to the Rust type (which isn't exposed via PyO3) but a
structure-compatible Python mirror, extended with a ``kind`` tag (``"list"``, ``"tuple"``,
``"namedtuple"``, ``"dict"``) recording the concrete Python container to reconstruct on
:meth:`~DataTree.to_python` -- something the Rust side has no need to track, since it only
distinguishes indexed vs. keyed children.

Any Python object may opt into being treated as a branch (rather than a leaf) by implementing
a ``__datatree__() -> DataTree`` method; :func:`from_python` defers to it when present.
:class:`.Tracer` is the only such type in this package: a structured tracer flattens into its
child tracers, so that ``build(shot_loop(...))`` and ``build(x)`` are the same code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["DataTree"]

_CONTAINER_KINDS = frozenset({"list", "tuple", "namedtuple", "dict"})


class DataTree:
    """A leaf value, or a branch of child :class:`DataTree`\\ s, some of which may be keyed.

    Construct via :meth:`leaf`, :meth:`branch`, or :func:`from_python`; the ``__init__``
    signature is considered private.
    """

    __slots__ = ("_children", "_key_index", "_keys", "_kind", "_meta", "_value")

    def __init__(
        self,
        kind: str,
        value: Any = None,
        children: Sequence[DataTree] = (),
        keys: Sequence[Any | None] | None = None,
        meta: Any = None,
    ) -> None:
        self._kind = kind
        self._value = value
        self._children = tuple(children)
        self._keys = tuple(keys) if keys is not None else (None,) * len(self._children)
        self._meta = meta
        self._key_index = {k: i for i, k in enumerate(self._keys) if k is not None}

    # -- construction -------------------------------------------------------------------

    @classmethod
    def leaf(cls, value: Any) -> DataTree:
        """A leaf :class:`DataTree` holding ``value``."""
        return cls("leaf", value=value)

    @classmethod
    def branch(
        cls,
        kind: str,
        children: Sequence[DataTree],
        keys: Sequence[Any | None] | None = None,
        meta: Any = None,
    ) -> DataTree:
        """A branch :class:`DataTree` of ``kind`` (one of ``"list"``, ``"tuple"``,
        ``"namedtuple"``, ``"dict"``) holding ``children``, optionally keyed."""
        if kind not in _CONTAINER_KINDS:
            raise ValueError(f"unknown DataTree branch kind {kind!r}")
        return cls(kind, children=children, keys=keys, meta=meta)

    # -- shape ------------------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """Whether this is a leaf (vs. a branch)."""
        return self._kind == "leaf"

    @property
    def kind(self) -> str:
        """``"leaf"``, or the branch kind (``"list"``, ``"tuple"``, ``"namedtuple"``, ``"dict"``)."""
        return self._kind

    @property
    def value(self) -> Any:
        """The leaf value. Raises :class:`TypeError` on a branch."""
        if not self.is_leaf:
            raise TypeError("value is only defined on a leaf DataTree")
        return self._value

    @property
    def children(self) -> tuple[DataTree, ...]:
        """The direct children of a branch, in order. Empty for a leaf."""
        return self._children

    @property
    def keys(self) -> tuple[Any | None, ...]:
        """One entry per child: its string key, or ``None`` if unkeyed."""
        return self._keys

    def __len__(self) -> int:
        return 1 if self.is_leaf else len(self._children)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataTree):
            return NotImplemented
        if self.is_leaf or other.is_leaf:
            return self.is_leaf and other.is_leaf and self._value == other._value
        return (
            self._kind == other._kind
            and self._keys == other._keys
            and self._children == other._children
        )

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"DataTree.leaf({self._value!r})"
        return f"DataTree.branch({self._kind!r}, {list(self.iter_children())!r})"

    # -- navigation -------------------------------------------------------------------------

    def iter_children(self) -> Iterable[tuple[Any | None, DataTree]]:
        """Iterate over ``(key_or_None, child)`` pairs, in order."""
        return zip(self._keys, self._children)

    def get(self, index: int) -> DataTree:
        """The child at position ``index``. Raises :class:`TypeError` on a leaf."""
        if self.is_leaf:
            raise TypeError("get() is only valid on a branch DataTree")
        return self._children[index]

    def get_by_key(self, key: Any) -> DataTree | None:
        """The child keyed by ``key``, or ``None`` if absent (or on a leaf)."""
        if self.is_leaf:
            return None
        index = self._key_index.get(key)
        return None if index is None else self._children[index]

    def get_by_path(self, path: Iterable[int | Any]) -> DataTree | None:
        """Walk ``path`` (a mix of positional indices and string keys), or ``None`` if any
        step is invalid (out-of-range index, unknown key, or descending through a leaf)."""
        node = self
        for entry in path:
            if node.is_leaf:
                return None
            if isinstance(entry, str):
                node = node.get_by_key(entry)
            elif 0 <= entry < len(node._children):
                node = node._children[entry]
            else:
                return None
            if node is None:
                return None
        return node

    # -- traversal --------------------------------------------------------------------------

    def leaves(self) -> list[Any]:
        """All leaf values, in depth-first order."""
        if self.is_leaf:
            return [self._value]
        out: list[Any] = []
        for child in self._children:
            out.extend(child.leaves())
        return out

    def paths(self) -> list[list[int | Any]]:
        """The raw path (list of index/key entries) to each leaf, in depth-first order.

        A bare leaf tree (no branch at all) yields a single empty path ``[]``.
        """
        if self.is_leaf:
            return [[]]
        out: list[list[int | Any]] = []
        for index, (key, child) in enumerate(self.iter_children()):
            entry = index if key is None else key
            out.extend([entry, *sub] for sub in child.paths())
        return out

    def dotted_paths(self) -> list[str]:
        """:meth:`paths`, rendered as dotted strings; a bare leaf renders as ``"out"``."""
        return [".".join(str(entry) for entry in path) if path else "out" for path in self.paths()]

    def num_leaves(self) -> int:
        """The total number of leaves."""
        return 1 if self.is_leaf else sum(child.num_leaves() for child in self._children)

    def map_leaves(self, f: Callable[[Any], Any]) -> DataTree:
        """A new tree with the same shape, replacing each leaf value with ``f(value)``."""
        if self.is_leaf:
            return DataTree.leaf(f(self._value))
        children = [child.map_leaves(f) for child in self._children]
        return DataTree.branch(self._kind, children, self._keys, self._meta)

    def unflatten(self, values: Iterable[Any]) -> DataTree:
        """A new tree with the same shape as ``self``, taking leaf values from ``values`` in
        depth-first order. Raises :class:`ValueError` if the count doesn't match."""
        values = list(values)
        expected = self.num_leaves()
        if len(values) != expected:
            raise ValueError(f"unflatten: expected {expected} values, got {len(values)}")
        it = iter(values)

        def build(template: DataTree) -> DataTree:
            if template.is_leaf:
                return DataTree.leaf(next(it))
            children = [build(child) for child in template._children]
            return DataTree.branch(template._kind, children, template._keys, template._meta)

        return build(self)

    # -- Python interop -----------------------------------------------------------------

    def to_python(self) -> Any:
        """Rebuild the Python object this tree was built from (or an equivalent structure)."""
        if self.is_leaf:
            return self._value
        children = [child.to_python() for child in self._children]
        if self._kind == "list":
            return children
        if self._kind == "tuple":
            return tuple(children)
        if self._kind == "namedtuple":
            return self._meta(*children)
        if self._kind == "dict":
            return dict(zip(self._keys, children))
        raise AssertionError(f"unhandled DataTree branch kind {self._kind!r}")


def from_python(obj: Any) -> DataTree:
    """Parse a Python object into a :class:`DataTree`.

    ``list``/``tuple``/``dict``/``namedtuple`` become branches (recursively); a ``dict``'s
    insertion order is preserved (not sorted). Any object with a ``__datatree__()`` method is
    deferred to instead. Everything else becomes a leaf.
    """
    if isinstance(obj, DataTree):
        return obj
    to_tree = getattr(obj, "__datatree__", None)
    if to_tree is not None:
        return to_tree()
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return DataTree.branch(
            "namedtuple",
            [from_python(value) for value in obj],
            keys=obj._fields,
            meta=type(obj),
        )
    if isinstance(obj, list):
        return DataTree.branch("list", [from_python(value) for value in obj])
    if isinstance(obj, tuple):
        return DataTree.branch("tuple", [from_python(value) for value in obj])
    if isinstance(obj, dict):
        return DataTree.branch(
            "dict", [from_python(value) for value in obj.values()], keys=list(obj.keys())
        )
    return DataTree.leaf(obj)


DataTree.from_python = staticmethod(from_python)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
