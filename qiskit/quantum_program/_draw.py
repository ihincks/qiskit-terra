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

"""A renderer for :class:`.Tracer` expression trees, shared by :meth:`.Tracer.draw` and
:meth:`.QuantumProgram.draw`.

Two independent renderings of the same underlying node/edge structure:

* ``output="text"`` (:func:`_render_blocks`) walks each output leaf and prints one indented
  ASCII tree per leaf; a node referenced more than once (fan-out/CSE) is tagged ``%n`` and
  expanded only at its first occurrence, so a tracer whose graph revisits the same node many
  times can't blow up the rendering.
* ``output="graphviz"`` (:func:`_render_graphviz`) instead lays out the *node graph itself* --
  one box per distinct ``_Node``, so a shared node is simply a box with more than one incoming
  edge, no ``%n`` tagging needed -- via the ``dot`` command-line tool, returning a
  :class:`PIL.Image.Image`. This requires the optional Graphviz and Pillow dependencies. Each
  edge (wire) is labelled with the destination argument name, the source port path if it comes
  from a multi-output node, and the wire's dtype/shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from qiskit.utils import optionals as _optionals

from ._tracer import _BINARY

__all__: list[str] = []


class _Text(str):
    """A ``str`` that reprs as itself (unquoted, multi-line), so a drawing renders directly in
    a REPL or notebook while remaining an ordinary string for tests/``print``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


def _op_display(node: _Node) -> str:
    """The op name plus any op-specific attributes, e.g. ``"mean(axis=0)"``."""
    op = node.op
    if op == "input":
        return f"input({node.attrs['key']!r})"
    if op == "constant":
        return "constant"
    if op == "shot_loop":
        return f"shot_loop(shots={node.attrs['shots']}, circuits={len(node.attrs['circuits'])})"
    if op == "parameter_expressions":
        return (
            f"parameter_expressions(expressions={len(node.attrs['expressions'])}, "
            f"parameters={len(node.attrs['parameters'])})"
        )
    parts = []
    if node.attrs.get("axis") is not None:
        parts.append(f"axis={node.attrs['axis']}")
    if node.attrs.get("ddof") is not None:
        parts.append(f"ddof={node.attrs['ddof']}")
    return f"{op}({', '.join(parts)})" if parts else op


def _path_suffix(path: list[int | str]) -> str:
    """A port path rendered as trailing index/key brackets, e.g. ``[0]["meas"]``."""
    return "".join(f"[{entry!r}]" for entry in path)


def _type_str(tracer: Tracer) -> str:
    return f"{tracer.dtype}{tracer.shape}"


def _constant_extra(node: _Node) -> str:
    """A short numeric preview trailing a ``constant`` node's line. The value is flattened
    first: ``array2string``'s ``threshold``/``edgeitems`` summarization only elides elements
    along an axis once that axis itself is long, so a large but "flat-ish" array (e.g. shape
    ``[2, 2, 5, 1]``) would otherwise print in full; flattening keeps the preview short (and
    bounded) regardless of the original shape, which is shown separately on the node's
    outgoing edge(s)."""
    flat = node.attrs["value"].reshape(-1)
    text = np.array2string(flat, precision=4, threshold=8, separator=", ")
    return f"  {text}"


def _arg_label(parent_op: str, index: int) -> str:
    """The label prefixed to a rendered arg, e.g. ``"x: "``, ``"params[0]: "``, or ``""``."""
    if parent_op in _BINARY:
        return "x: " if index == 0 else "y: "
    if parent_op == "shot_loop":
        return f"params[{index}]: "
    if parent_op == "parameter_expressions":
        return "values: "
    return ""


def _render_blocks(tree: DataTree) -> list[str]:
    """One rendered, indented expression tree per leaf of ``tree``."""
    leaves: list[Tracer] = tree.leaves()
    keys = tree.dotted_paths()

    counts: dict[int, int] = {}

    def count(tracer: Tracer) -> None:
        node_id = id(tracer._node)  # pylint: disable=protected-access
        counts[node_id] = counts.get(node_id, 0) + 1
        if counts[node_id] == 1:
            for arg in tracer._node.args:  # pylint: disable=protected-access
                count(arg)

    for leaf in leaves:
        count(leaf)

    tags: dict[int, int] = {}
    expanded: set[int] = set()

    def render(tracer: Tracer, prefix: str, connector: str, label: str, lines: list[str]) -> None:
        node = tracer._node  # pylint: disable=protected-access
        path = tracer._path  # pylint: disable=protected-access
        node_id = id(node)

        tag = ""
        if counts[node_id] > 1:
            if node_id not in tags:
                tags[node_id] = len(tags) + 1
            tag = f"%{tags[node_id]} "

        extra = _constant_extra(node) if node.op == "constant" else ""
        header = f"{label}{tag}{_op_display(node)}{_path_suffix(path)} → {_type_str(tracer)}{extra}"
        lines.append(f"{prefix}{connector}{header}")

        if counts[node_id] > 1:
            if node_id in expanded:
                return
            expanded.add(node_id)

        if connector == "":
            child_prefix = prefix
        elif connector == "└─ ":
            child_prefix = prefix + "   "
        else:
            child_prefix = prefix + "│  "

        args = node.args
        for index, arg in enumerate(args):
            is_last = index == len(args) - 1
            render(
                arg,
                child_prefix,
                "└─ " if is_last else "├─ ",
                _arg_label(node.op, index),
                lines,
            )

    blocks: list[str] = []
    for key, leaf in zip(keys, leaves):
        lines: list[str] = []
        render(leaf, "", "", f"{key}: ", lines)
        blocks.append("\n".join(lines))
    return blocks


def _dot_id(node: _Node) -> str:
    """A dot-safe node identifier for ``node``, stable for the lifetime of one render call."""
    return f"n{id(node)}"


def _dot_quote(text: str) -> str:
    """Quote ``text`` as a dot string literal, escaping backslashes/quotes and real newlines."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _node_label(node: _Node) -> str:
    """The node-type label: op name plus attributes, e.g. ``"mean(axis=0)"``."""
    label = _op_display(node)
    if node.op == "constant":
        label += _constant_extra(node)
    return label


def _edge_label(parent_op: str, index: int, arg: Tracer) -> str:
    """The label on the edge feeding argument ``index`` of ``parent_op``: an arrow from the
    source-side port path (if ``arg`` is a port into a multi-output node, e.g. ``[0]["meas"]``)
    to the destination-side argument name (if ``parent_op`` names its args, e.g. ``"x"``,
    ``"params[0]"``) -- one, both, or neither may be present -- plus the wire's dtype/shape."""
    name = _arg_label(parent_op, index).removesuffix(": ")
    path = _path_suffix(arg._path)  # pylint: disable=protected-access
    if name and path:
        connector = f"{path} → {name}"
    elif path:
        connector = f"{path} → ·"
    elif name:
        connector = f"· → {name}"
    else:
        connector = ""
    lines = [part for part in (connector, _type_str(arg)) if part]
    return "\n".join(lines)


def _render_graphviz(tree: DataTree) -> Image:
    """Lay out the node graph reachable from every leaf of ``tree`` via Graphviz, returning a
    :class:`PIL.Image.Image`: one box per distinct node, one labelled edge per argument, and a
    shaded sink box per output leaf."""
    leaves: list[Tracer] = tree.leaves()
    keys = tree.dotted_paths()

    seen: dict[int, _Node] = {}  # id(_Node) -> node; also pins lifetimes vs. id() recycling
    order: list[_Node] = []
    edges: list[tuple[str, str, str]] = []

    def visit(node: _Node) -> None:
        if id(node) in seen:
            return
        seen[id(node)] = node
        order.append(node)
        for index, arg in enumerate(node.args):
            visit(arg._node)  # pylint: disable=protected-access
            edges.append(
                (_dot_id(arg._node), _dot_id(node), _edge_label(node.op, index, arg))
            )  # pylint: disable=protected-access

    for leaf in leaves:
        visit(leaf._node)  # pylint: disable=protected-access

    lines = [
        "digraph quantum_program {",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for node in order:
        shape = "ellipse" if node.op in ("input", "constant") else "box"
        lines.append(f"  {_dot_id(node)} [shape={shape}, label={_dot_quote(_node_label(node))}];")
    for from_id, to_id, label in edges:
        attrs = f" [label={_dot_quote(label)}]" if label else ""
        lines.append(f"  {from_id} -> {to_id}{attrs};")

    for index, (key, leaf) in enumerate(zip(keys, leaves)):
        sink_id = f"out{index}"
        sink_label = f"{key}\n{_type_str(leaf)}"
        lines.append(
            f"  {sink_id} [shape=box, style=filled, fillcolor=lightgrey, "
            f"label={_dot_quote(sink_label)}];"
        )
        suffix = _path_suffix(leaf._path)  # pylint: disable=protected-access
        attrs = f" [label={_dot_quote(suffix)}]" if suffix else ""
        lines.append(
            f"  {_dot_id(leaf._node)} -> {sink_id}{attrs};"
        )  # pylint: disable=protected-access
    lines.append("}")

    return _run_dot("\n".join(lines))


@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PIL.require_in_call
def _run_dot(dot_str: str) -> Image:
    """Render a dot-language ``dot_str`` to a :class:`PIL.Image.Image` via the ``dot`` binary."""
    import io
    import subprocess

    from PIL import Image as _PILImage

    result = subprocess.run(
        ["dot", "-T", "png"], input=dot_str.encode("utf-8"), capture_output=True, check=True
    )
    return _PILImage.open(io.BytesIO(result.stdout))


def draw_tracers(tree: DataTree, output: Literal["text", "graphviz"] = "text") -> _Text | Image:
    """Render every leaf of ``tree`` (a :class:`.DataTree` of :class:`.Tracer`\\ s).

    ``output="text"`` (the default) returns a multi-line string: one indented expression tree
    per leaf, separated by a blank line. ``output="graphviz"`` instead lays out the underlying
    node graph itself (one box per node, shared nodes naturally have more than one incoming
    edge) as an image, to be viewed rather than printed; this requires the optional Graphviz
    and Pillow dependencies.
    """
    if output == "text":
        return _Text("\n\n".join(_render_blocks(tree)))
    if output == "graphviz":
        return _render_graphviz(tree)
    raise ValueError(f"unknown draw() output {output!r}; expected 'text' or 'graphviz'")


if TYPE_CHECKING:
    from PIL.Image import Image

    from ._datatree import DataTree
    from ._tracer import Tracer, _Node
