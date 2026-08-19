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

"""Tests for the public surface of qiskit.quantum_program."""

import ast
import inspect
import operator
import pathlib
from collections import OrderedDict, namedtuple

import numpy as np

import qiskit.quantum_program
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, QuantumRegister
from qiskit.quantum_program import (
    DataTree,
    DType,
    TensorType,
    Tracer,
    add,
    bind_parameters,
    bit,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    bounded,
    broadcast_to,
    build,
    cast,
    constant,
    divide,
    f32,
    f64,
    i64,
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
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestTypeStub(QiskitTestCase):
    """Tests that pyi stubs don't drift from reality."""

    @staticmethod
    def stub() -> ast.Module:
        """The parsed stub."""
        source = pathlib.Path(inspect.getfile(qiskit.quantum_program)).with_suffix(".pyi")
        return ast.parse(source.read_text())

    @classmethod
    def declarations(cls) -> dict[str, set[str]]:
        """The members the stub declares, for each class it declares."""
        return {
            declaration.name: {
                node.name if isinstance(node, ast.FunctionDef) else node.target.id
                for node in declaration.body
                if isinstance(node, (ast.FunctionDef, ast.AnnAssign))
            }
            for declaration in cls.stub().body
            if isinstance(declaration, ast.ClassDef)
        }

    @classmethod
    def exports(cls) -> set[str]:
        """Every name the stub gives the module, declared here or imported from a sibling."""
        names = set()
        for node in cls.stub().body:
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.AnnAssign):
                names.add(node.target.id)
            # A name written in Python is imported from the module defining it, so that a type
            # checker reads its real signature. Anything else imported here is a type used below.
            elif isinstance(node, ast.ImportFrom) and node.level:
                names.update(alias.asname or alias.name for alias in node.names)
        return names

    def test_stub_covers_public_module(self):
        """Test that everything the module exports is declared in the stub."""
        self.assertEqual(self.exports(), set(qiskit.quantum_program.__all__))

    def test_stub_declares_class_members(self):
        """Test that the stub declares exactly the members each class has."""
        # Written out rather than filtered, so that a new member is a deliberate edit here. These
        # are the names pyo3 adds of its own accord, alongside those the class declares.
        generated = {"__doc__", "__ge__", "__gt__", "__le__", "__lt__", "__module__", "__ne__"}
        for name, declared in self.declarations().items():
            with self.subTest(name):
                # A stub spells the constructor `__init__` where a pyclass exposes `__new__`.
                declared = {"__new__" if member == "__init__" else member for member in declared}
                members = {
                    member
                    for member in vars(getattr(qiskit.quantum_program, name))
                    # A stub declares no private member.
                    if member.startswith("__") or not member.startswith("_")
                }
                self.assertEqual(declared, members - generated)


class TestDataTree(QiskitTestCase):
    """Tests for DataTree, the container a program's inputs and outputs travel in."""

    def test_list_parses_to_unnamed_branch(self):
        """Test that a list becomes a branch addressed only by position."""
        tree = DataTree([1, 2])
        self.assertEqual(len(tree), 2)
        self.assertEqual([tree[0], tree[1]], [1, 2])

    def test_tuple_parses_like_a_list(self):
        """Test that a tuple parses the same way a list does."""
        self.assertEqual(DataTree((1, 2)), DataTree([1, 2]))

    def test_dict_parses_to_named_branch(self):
        """Test that a dict becomes a branch with its keys as names."""
        tree = DataTree({"counts": 3, "shots": 100})
        self.assertEqual(tree["counts"], 3)
        self.assertEqual(tree["shots"], 100)

    def test_dict_keeps_insertion_order(self):
        """Test that names appear in the order the dict was written, not sorted."""
        self.assertEqual(DataTree({"z": 1, "a": 2, "m": 3}).keys(), ["z", "a", "m"])
        self.assertEqual(DataTree(OrderedDict([("z", 1), ("a", 2)])).keys(), ["z", "a"])

    def test_namedtuple_parses_to_named_branch(self):
        """Test that a namedtuple is named by its fields, not caught by the tuple case."""
        point = namedtuple("Point", ["x", "y"])
        self.assertEqual(DataTree(point(1, 2)), DataTree({"x": 1, "y": 2}))

    def test_malformed_namedtuple_refused(self):
        """Test that fields failing to name each item exactly once are refused."""

        class TooFew(tuple):
            """A tuple naming fewer fields than it holds items."""

            _fields = ("x",)

        class Repeated(tuple):
            """A tuple naming one field twice."""

            _fields = ("x", "x")

        for malformed in (TooFew((1, 2)), Repeated((1, 2))):
            with self.assertRaisesRegex(ValueError, "does not name each of the 2 items"):
                DataTree(malformed)

    def test_nesting_parses_all_the_way_down(self):
        """Test that containers inside containers each become a branch."""
        tree = DataTree({"a": [1, {"b": 2}]})
        self.assertEqual(tree["a"][0], 1)
        self.assertEqual(tree["a"][1]["b"], 2)

    def test_other_objects_become_leaves(self):
        """Test that a leaf holds whatever object it was given, of whatever type."""
        array = np.zeros(2)
        tree = DataTree({"array": array, "text": "abc", "none": None})
        self.assertIs(tree["array"], array)
        self.assertEqual(tree["text"], "abc")
        self.assertIsNone(tree["none"])

    def test_parsing_a_tree_is_idempotent(self):
        """Test that a tree parses to an equal tree."""
        tree = DataTree({"a": [1, 2]})
        self.assertEqual(DataTree(tree), tree)

    def test_leaf_of_holds_a_value_unparsed(self):
        """Test that a value a constructor would decompose can be held as one leaf."""
        tree = DataTree.leaf_of([1, 2])
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf, [1, 2])

    def test_hook_decomposes_an_object(self):
        """Test that __datatree__ is used in place of treating the object as a leaf."""

        class Port:
            """An object that describes itself as a branch."""

            def __datatree__(self):
                return DataTree({"c0": 1, "c1": 2})

        self.assertEqual(DataTree(Port()), DataTree({"c0": 1, "c1": 2}))
        self.assertEqual(DataTree({"port": Port()}), DataTree({"port": {"c0": 1, "c1": 2}}))

    def test_hook_must_return_a_tree(self):
        """Test that a hook returning anything but a DataTree is an error."""

        class Sloppy:
            """An object whose hook returns a dict."""

            def __datatree__(self):
                return {"a": 1}

        with self.assertRaisesRegex(TypeError, "__datatree__ returned dict"):
            DataTree(Sloppy())

    def test_hook_looked_up_on_the_type(self):
        """Test that the hook is found on the type rather than on the instance."""

        class Proxy:
            """An object that answers any attribute."""

            def __getattr__(self, name):
                return 7

        class Port:
            """An object that describes itself as a branch."""

            def __datatree__(self):
                return DataTree({"a": 1})

        self.assertTrue(DataTree(Proxy()).is_leaf)
        self.assertTrue(DataTree(Port).is_leaf)

    def test_child_addressed_by_name_or_position(self):
        """Test that the same child answers to its name and to where it sits."""
        tree = DataTree({"a": 1, "b": 2})
        self.assertEqual(tree["b"], tree[1])
        self.assertEqual(tree[-1], 2)

    def test_reading_a_child_unwraps_a_leaf(self):
        """Test that reading a branch gives a tree and reading a leaf gives its value."""
        tree = DataTree({"counts": [3, 4]})
        self.assertEqual(tree["counts"], DataTree([3, 4]))
        self.assertEqual(tree["counts"][0], 3)

    def test_value_addressed_by_dotted_path(self):
        """Test that a path of names and positions reaches a value below a child."""
        tree = DataTree({"counts": [3, {"creg": 7}], "ev": 0.1})
        self.assertEqual(tree["counts.0"], 3)
        self.assertEqual(tree["counts.1.creg"], 7)
        self.assertEqual(tree["0.1.creg"], 7)
        self.assertEqual(tree["counts.1"], DataTree({"creg": 7}))
        self.assertIn("counts.1.creg", tree)
        self.assertNotIn("counts.2", tree)

    def test_bad_address_raises(self):
        """Test that a missing name, an out-of-range position and an unusable key each raise."""
        tree = DataTree({"a": 1})
        with self.assertRaises(KeyError):
            _ = tree["b"]
        with self.assertRaises(IndexError):
            _ = tree[1]
        with self.assertRaisesRegex(TypeError, "not by float"):
            _ = tree[1.5]
        with self.assertRaises(KeyError):
            _ = tree[""]

    def test_contains_asks_about_addresses(self):
        """Test that `in` reads its argument the way indexing does."""
        tree = DataTree({"a": np.zeros(2)})
        self.assertIn("a", tree)
        self.assertIn(0, tree)
        self.assertNotIn("b", tree)
        self.assertNotIn(1, tree)

    def test_length_and_iteration(self):
        """Test that length and iteration cover the children, in order."""
        tree = DataTree({"a": 1, "b": [2, 3]})
        self.assertEqual(len(tree), 2)
        self.assertEqual(list(tree), [1, DataTree([2, 3])])

    def test_leaf_has_no_children(self):
        """Test that a leaf holds no children and its value is read through `leaf`."""
        tree = DataTree(5)
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf, 5)
        self.assertEqual(len(tree), 0)
        self.assertEqual(list(tree), [])

    def test_branch_has_no_leaf_value(self):
        """Test that a branch raises when asked for a leaf value."""
        tree = DataTree([1])
        self.assertFalse(tree.is_leaf)
        with self.assertRaisesRegex(TypeError, "no leaf value"):
            _ = tree.leaf

    def test_named_branch_is_a_mapping(self):
        """Test that a branch naming its children converts to a dict."""
        tree = DataTree({"a": 1, "b": 2})
        self.assertTrue(tree.is_mapping)
        self.assertEqual(tree.keys(), ["a", "b"])
        self.assertEqual(tree.values(), [1, 2])
        self.assertEqual(tree.items(), [("a", 1), ("b", 2)])
        self.assertEqual(dict(tree), {"a": 1, "b": 2})
        self.assertEqual(dict(**tree), {"a": 1, "b": 2})

    def test_unnamed_branch_is_not_a_mapping(self):
        """Test that a branch naming none of its children has no mapping form."""
        tree = DataTree([1, 2])
        self.assertFalse(tree.is_mapping)
        for access in (tree.keys, tree.values, tree.items):
            with self.assertRaisesRegex(TypeError, "unnamed children"):
                access()
        with self.assertRaises(TypeError):
            dict(tree)

    def test_leaf_is_not_a_mapping(self):
        """Test that a leaf has no children to name, and so no mapping form."""
        self.assertFalse(DataTree(5).is_mapping)
        with self.assertRaisesRegex(TypeError, "a leaf of a data tree has no mapping form"):
            dict(DataTree(5))

    def test_empty_branch_is_an_empty_mapping(self):
        """Test that an empty branch converts to an empty dict."""
        self.assertTrue(DataTree([]).is_mapping)
        self.assertEqual(dict(DataTree([])), {})

    def test_unaddressable_name_refused(self):
        """Test that a name with a dot, one of all digits, and a non-string are refused."""
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            DataTree({"a.b": 1})
        with self.assertRaisesRegex(ValueError, "only of digits"):
            DataTree({"12": 1})
        with self.assertRaisesRegex(TypeError, "not with int"):
            DataTree({1: 2})

    def test_equality_compares_structure_and_leaves(self):
        """Test that trees are equal when built the same way with equal leaves."""
        self.assertEqual(DataTree({"a": [1, 2]}), DataTree({"a": [1, 2]}))
        self.assertNotEqual(DataTree({"a": [1, 2]}), DataTree({"a": [1, 3]}))
        self.assertNotEqual(DataTree({"a": [1, 2]}), DataTree({"a": [1, 2, 3]}))
        self.assertNotEqual(DataTree([1]), 1)

    def test_naming_affects_equality(self):
        """Test that trees differing only in which children are named are unequal."""
        self.assertNotEqual(DataTree({"a": 1}), DataTree([1]))
        self.assertNotEqual(DataTree({"a": 1}), DataTree({"b": 1}))
        self.assertNotEqual(DataTree(1), DataTree([1]))

    def test_identical_leaves_skip_comparison(self):
        """Test that a tree whose leaves are arrays can be compared against itself."""
        array = np.zeros(2)
        self.assertEqual(DataTree({"a": array}), DataTree({"a": array}))

    def test_tree_is_unhashable(self):
        """Test that a tree is unhashable, since its leaves need not be."""
        with self.assertRaises(TypeError):
            hash(DataTree([1]))

    def test_repr(self):
        """Test that repr renders a leaf by its own repr and a branch in brackets."""
        self.assertEqual(repr(DataTree(5)), "DataTree(5)")
        self.assertEqual(
            repr(DataTree({"counts": [3, 4], "shots": 100})),
            "DataTree([counts: [3, 4], shots: 100])",
        )


class TestTensorTypes(QiskitTestCase):
    """Tests for dtypes and the tensor types they build."""

    def test_lowercase_alias_is_the_dtype(self):
        """Test that a lower-case alias is the dtype it spells."""
        self.assertEqual(f64, DType.F64)
        self.assertEqual(bit, DType.Bit)

    def test_indexing_a_dtype_builds_a_type(self):
        """Test that indexing a dtype with a shape gives that type."""
        self.assertEqual(f64[3], TensorType(DType.F64, (3,)))
        self.assertEqual(f64[3, 4].shape, (3, 4))
        self.assertEqual(f64[()].shape, ())

    def test_type_carries_dtype_and_shape(self):
        """Test that a type reports what it was built from."""
        self.assertEqual(f64[3, 4].dtype, DType.F64)
        self.assertEqual(TensorType(DType.Bit, [2]).shape, (2,))

    def test_bounded_axis(self):
        """Test that a bounded axis has its maximum and renders as one."""
        self.assertEqual(bit[1024, bounded(64)].shape, (1024, bounded(64)))
        self.assertEqual(str(bit[1024, bounded(64)]), "Bit[1024, <=64]")
        self.assertEqual(bounded(64).max, 64)
        self.assertEqual(repr(bounded(64)), "bounded(64)")

    def test_equal_types_are_one_key(self):
        """Test that types of equal dtype and shape are equal and hash alike."""
        self.assertEqual({f64[3], TensorType(DType.F64, [3])}, {f64[3]})
        self.assertNotEqual(f64[3], f32[3])
        self.assertNotEqual(f64[3], f64[3, 1])

    def test_type_renders_dtype_and_shape(self):
        """Test that a type renders as its dtype over its shape."""
        self.assertEqual(repr(f64[3]), "TensorType(F64[3])")
        self.assertEqual(str(f64[3]), "F64[3]")

    def test_unusable_axis_refused(self):
        """Test that an axis sized by anything but an integer or a bound raises."""
        with self.assertRaisesRegex(TypeError, "not by float"):
            _ = f64[1.5]


class TestTracer(QiskitTestCase):
    """Tests for writing expressions with tracers."""

    def test_input_carries_its_declared_type(self):
        """Test that a declared input reports the type it was declared with."""
        x = qp_input("x", f64[3])
        self.assertEqual(x.type, f64[3])
        self.assertEqual(x.dtype, DType.F64)
        self.assertEqual(x.shape, (3,))

    def test_arithmetic_infers_its_type(self):
        """Test that every arithmetic operator reports the type it produces."""
        x = qp_input("x", f64[3])
        y = qp_input("y", f64[1])
        for value in (x + y, x - y, x * y, x / y, x % y, x**y):
            self.assertEqual(value.type, f64[3])

    def test_operands_are_promoted_and_broadcast(self):
        """Test that operands of different dtypes and shapes combine."""
        self.assertEqual((qp_input("x", f64[3]) + qp_input("y", i64[3])).type, f64[3])
        self.assertEqual((qp_input("x", f64[2, 3]) * qp_input("y", f64[1, 3])).type, f64[2, 3])
        self.assertEqual((qp_input("b", bit[3]) + qp_input("y", f64[3])).type, f64[3])

    def test_bitwise_operators(self):
        """Test that the bitwise operators produce bits."""
        b = qp_input("b", bit[2, 3])
        c = qp_input("c", bit[2, 3])
        for value in (b & c, b | c, b ^ c, ~b):
            self.assertEqual(value.type, bit[2, 3])

    def test_reductions_remove_their_axis(self):
        """Test that each reduction drops the axis it folds along."""
        x = qp_input("x", f64[4, 2])
        b = qp_input("b", bit[4, 2])
        self.assertEqual(x.mean(0).type, f64[2])
        self.assertEqual(x.var(1).type, f64[4])
        self.assertEqual(x.std(0, ddof=1).type, f64[2])
        self.assertEqual(b.parity(0).type, bit[2])
        self.assertEqual(b.mean(0).type, f64[2])

    def test_other_operand_becomes_a_constant(self):
        """Test that an operand that is not a tracer is carried as a constant."""
        x = qp_input("x", f64[3])
        self.assertEqual((x + 1.0).type, f64[3])
        self.assertEqual(constant([1.0, 2.0]).type, f64[2])

    def test_numpy_defers_to_the_tracer(self):
        """Test that an array or a numpy scalar on the left builds a node."""
        x = qp_input("x", f64[3])
        self.assertIsInstance(np.float64(2.0) * x, Tracer)
        self.assertIsInstance(np.zeros(3) + x, Tracer)

    def test_shape_mistake_raises_where_written(self):
        """Test that operands of incompatible shapes raise at the operation."""
        with self.assertRaisesRegex(ValueError, r"qiskit.add: shapes \[3\] and \[4\]"):
            _ = qp_input("x", f64[3]) + qp_input("y", f64[4])

    def test_dtype_mistake_raises_where_written(self):
        """Test that an operand of an unsupported dtype raises at the operation."""
        with self.assertRaisesRegex(ValueError, "qiskit.bitwise_and"):
            _ = qp_input("b", bit[3]) & qp_input("x", f64[3])

    def test_cast_and_broadcast(self):
        """Test that a value can be given another dtype or a wider shape."""
        x = qp_input("x", i64[3])
        self.assertEqual(cast(x, DType.F64).type, f64[3])
        self.assertEqual(broadcast_to(x, (2, 3)).type, i64[2, 3])

    def test_repr_names_what_produced_it(self):
        """Test that a tracer renders its operation and its type."""
        x = qp_input("x", f64[3])
        self.assertEqual(repr(x), "Tracer(input 'x', F64[3])")
        self.assertEqual(repr(x + x), "Tracer(qiskit.add, F64[3])")


class TestStandaloneOps(QiskitTestCase):
    """Tests that every operator has an interchangeable function."""

    @staticmethod
    def program(expression, type_):
        """A program applying `expression` to two inputs of `type_`."""
        x = qp_input("x", type_)
        y = qp_input("y", type_)
        return build({"z": expression(x, y)})

    def compare(self, function, form, type_, **arguments):
        """Assert that `function` and `form` build one program and produce one value."""
        function, form = self.program(function, type_), self.program(form, type_)
        self.assertEqual(function._node_type_counts(), form._node_type_counts())
        # An expression reaching one input declares one, so the other argument is dropped.
        arguments = {name: arguments[name] for name in function.input_types().keys()}
        np.testing.assert_array_equal(function(**arguments)["z"], form(**arguments)["z"])

    def test_arithmetic_functions_match_operators(self):
        """Test that each arithmetic function builds and computes what its operator does."""
        for function, form in (
            (add, operator.add),
            (subtract, operator.sub),
            (multiply, operator.mul),
            (divide, operator.truediv),
            (remainder, operator.mod),
            (power, operator.pow),
        ):
            with self.subTest(function.__name__):
                self.compare(function, form, f64[2], x=[8.0, 5.0], y=[3.0, 2.0])

    def test_bitwise_functions_match_operators(self):
        """Test that each bitwise function builds and computes what its operator does."""
        bits = {"x": np.array([True, True]), "y": np.array([True, False])}
        for function, form in (
            (bitwise_and, operator.and_),
            (bitwise_or, operator.or_),
            (bitwise_xor, operator.xor),
        ):
            with self.subTest(function.__name__):
                self.compare(function, form, bit[2], **bits)
        self.compare(lambda x, _y: bitwise_not(x), lambda x, _y: ~x, bit[2], **bits)

    def test_reduction_functions_match_methods(self):
        """Test that each reduction function builds and computes what its method does."""
        floats = {"x": np.array([[8.0, 5.0], [3.0, 2.0]]), "y": np.zeros((2, 2))}
        bits = {"x": np.array([[True, True], [True, False]]), "y": np.zeros((2, 2), dtype=bool)}
        for function, form, type_, arguments in (
            (lambda x, _y: mean(x, 0), lambda x, _y: x.mean(0), f64[2, 2], floats),
            (lambda x, _y: var(x, 1, 1.0), lambda x, _y: x.var(1, 1.0), f64[2, 2], floats),
            (lambda x, _y: std(x, 0), lambda x, _y: x.std(0), f64[2, 2], floats),
            (lambda x, _y: parity(x, 0), lambda x, _y: x.parity(0), bit[2, 2], bits),
        ):
            self.compare(function, form, type_, **arguments)

    def test_reflected_operators_keep_operand_order(self):
        """Test that an operand written on the left is the left operand."""
        x = qp_input("x", f64[1])
        for expression, expected in (
            (2.0 - x, [1.5]),
            (2.0 / x, [4.0]),
            (2.0**x, [2.0**0.5]),
            (1.5 % x, [0.0]),
        ):
            np.testing.assert_allclose(build(expression)(x=[0.5]).leaf, expected)


class TestBuild(QiskitTestCase):
    """Tests for turning expressions into a program."""

    def test_value_used_twice_becomes_one_node(self):
        """Test that a shared subexpression is built once."""
        x = qp_input("x", f64[3])
        squared = x * x
        program = build({"a": squared + 1.0, "b": squared - 1.0})
        self.assertEqual(program._node_type_counts()["qiskit.multiply"], 1)

    def test_equal_expressions_are_two_nodes(self):
        """Test that two expressions built alike are two nodes, since sharing is by identity."""
        x = qp_input("x", f64[3])
        program = build({"a": x * x, "b": x * x})
        self.assertEqual(program._node_type_counts()["qiskit.multiply"], 2)

    def test_one_value_may_be_two_outputs(self):
        """Test that naming one value twice gives two results from one node."""
        value = qp_input("x", f64[2]).mean(0)
        program = build({"a": value, "b": value})
        self.assertEqual(
            program._node_type_counts(),
            {"qiskit.parameter": 1, "qiskit.mean": 1, "qiskit.result": 2},
        )
        self.assertEqual(program(x=[1.0, 3.0])["b"], 2.0)

    def test_chain_longer_than_the_recursion_limit_builds(self):
        """Test that a chain deeper than Python's stack allows is built."""
        value = qp_input("x", f64[1])
        for _ in range(2000):
            value = value + 1.0
        self.assertEqual(build(value)(x=[0.0]).leaf, 2000.0)

    def test_unused_value_is_not_built(self):
        """Test that a value no output reaches is left out."""
        x = qp_input("x", f64[3])
        _ = x.mean(0)
        self.assertNotIn("qiskit.mean", build({"a": x + 1.0})._node_type_counts())

    def test_coercion_inserts_nothing(self):
        """Test that promoting and broadcasting add no nodes of their own."""
        program = build({"z": qp_input("x", f64[2, 3]) + qp_input("y", i64[3])})
        self.assertEqual(
            program._node_type_counts(),
            {"qiskit.parameter": 2, "qiskit.add": 1, "qiskit.result": 1},
        )

    def test_outputs_may_be_one_value(self):
        """Test that a single value builds a program whose output is a bare leaf."""
        program = build(qp_input("x", f64[3]).mean(0))
        self.assertEqual(program.output_types(), DataTree(f64[()]))
        self.assertEqual(program(x=[1.0, 2.0, 3.0]).leaf, 2.0)

    def test_outputs_may_nest(self):
        """Test that outputs come back in the structure they were declared in."""
        x = qp_input("x", f64[2])
        program = build({"pair": [x.mean(0), x.var(0)], "raw": x})
        self.assertEqual(
            program.output_types(),
            DataTree({"pair": [f64[()], f64[()]], "raw": f64[2]}),
        )
        results = program(x=[1.0, 3.0])
        self.assertEqual(results["pair"][0], 2.0)
        np.testing.assert_allclose(results["raw"], [1.0, 3.0])

    def test_inputs_are_declared_in_the_order_they_are_reached(self):
        """Test that the input structure follows the walk over the outputs."""
        a = qp_input("a", f64[1])
        b = qp_input("b", bit[2])
        self.assertEqual(build({"z": a.mean(0)}).input_types(), DataTree({"a": f64[1]}))
        self.assertEqual(build({"z": a + b}).input_types().keys(), ["a", "b"])
        self.assertEqual(build({"z": b + a}).input_types().keys(), ["b", "a"])

    def test_input_declared_twice_refused(self):
        """Test that two inputs of one name are refused, naming it."""
        with self.assertRaisesRegex(ValueError, "'x' is declared twice"):
            build({"z": qp_input("x", f64[1]) + qp_input("x", f64[1])})

    def test_output_that_is_not_a_value_refused(self):
        """Test that an output leaf that is not a tracer is refused, naming its path."""
        with self.assertRaisesRegex(TypeError, "output 'a.0' is a float"):
            build({"a": [1.0]})

    def test_unaddressable_output_name_refused(self):
        """Test that an output name a path could not address is refused."""
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            build({"a.b": qp_input("x", f64[1])})


class TestCallingAProgram(QiskitTestCase):
    """Tests for evaluating a program in process."""

    def test_arithmetic_runs(self):
        """Test that a program of arithmetic produces the values it describes."""
        x = qp_input("x", f64[4])
        program = build({"mean": x.mean(0), "shifted": x - 1.0})
        results = program(x=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual(results["mean"], 2.5)
        np.testing.assert_allclose(results["shifted"], [0.0, 1.0, 2.0, 3.0])

    def test_array_likes_are_accepted(self):
        """Test that anything reading as an array of the declared dtype is accepted."""
        program = build(qp_input("x", f64[2]).mean(0))
        for argument in ([1.0, 3.0], (1.0, 3.0), np.array([1.0, 3.0])):
            self.assertEqual(program(x=argument).leaf, 2.0)

    def test_input_of_another_type_refused(self):
        """Test that an input whose type is not the declared one is refused, naming both."""
        program = build(qp_input("x", f64[3]).mean(0))
        with self.assertRaisesRegex(ValueError, r"input 'x': expected F64\[3\], got I64\[3\]"):
            program(x=[1, 2, 3])
        with self.assertRaisesRegex(ValueError, r"expected F64\[3\], got F64\[2\]"):
            program(x=[1.0, 2.0])

    def test_missing_and_unexpected_keywords_reported_together(self):
        """Test that a call naming the wrong inputs reports both sets."""
        program = build({"z": qp_input("x", f64[1]) + qp_input("y", f64[1])})
        with self.assertRaisesRegex(
            TypeError, r"takes inputs \['x', 'y'\]; missing \['y'\], unexpected \['q'\]"
        ):
            program(x=[1.0], q=[1.0])

    def test_bits_cross_as_bools(self):
        """Test that a bit input and a bit output are arrays of bool."""
        b = qp_input("b", bit[2, 2])
        program = build({"parity": b.parity(1)})
        results = program(b=np.array([[True, False], [True, True]]))
        np.testing.assert_array_equal(results["parity"], [True, False])

    def test_division_by_zero_is_not_finite(self):
        """Test that a zero divisor gives a non-finite value rather than an error."""
        program = build(qp_input("x", f64[2]) / constant([0.0, 2.0]))
        np.testing.assert_array_equal(np.isfinite(program(x=[1.0, 4.0]).leaf), [False, True])

    def test_integer_arithmetic_has_no_undefined_case(self):
        """Test that an integer zero divisor gives zero and an overflow wraps."""
        x = qp_input("x", i64[2])
        divisors = constant(np.array([0, 2]))
        program = build({"quotient": x / divisors, "remainder": x % divisors})
        results = program(x=np.array([7, 7]))
        np.testing.assert_array_equal(results["quotient"], [0, 3])
        np.testing.assert_array_equal(results["remainder"], [0, 1])
        wrapped = build(x ** constant(np.array([100, 2])))(x=np.array([2, 3]))
        np.testing.assert_array_equal(wrapped.leaf, [0, 9])

    def test_negative_integer_exponent_refused(self):
        """Test that raising an integer to a negative power reports it rather than aborting."""
        program = build(qp_input("x", i64[1]) ** constant(np.array([-1])))
        with self.assertRaisesRegex(ValueError, "exponent of dtype I64 cannot be negative"):
            program(x=np.array([2]))

    def test_repr_shows_the_declared_structures(self):
        """Test that a program renders the structures it declares."""
        program = build({"z": qp_input("x", f64[1])})
        self.assertEqual(repr(program), "QuantumProgram(inputs=[x: _], outputs=[z: _])")


class TestCircuits(QiskitTestCase):
    """Tests for wiring circuits and their parameters into a program."""

    @staticmethod
    def circuit(parameters=(), registers=(("c", 2),)):
        """A two-qubit circuit rotating by each of `parameters` and measuring into `registers`.

        Args:
            parameters: The parameters to rotate by, one instruction each.
            registers: The classical registers to measure into, as `(name, width)` pairs.

        Returns:
            The circuit.
        """
        cregs = [ClassicalRegister(width, name) for name, width in registers]
        qubits = QuantumRegister(max([2, *(len(register) for register in cregs)]), "q")
        circuit = QuantumCircuit(qubits, *cregs)
        for index, parameter in enumerate(parameters):
            circuit.rx(parameter, index % 2)
        for register in cregs:
            circuit.measure(qubits[: len(register)], register)
        return circuit

    def test_outcomes_are_a_mapping_per_circuit(self):
        """Test that a shot loop gives one entry per circuit, keyed by register name."""
        first = self.circuit(registers=(("a", 1), ("b", 2)))
        second = self.circuit(registers=(("c", 2),))
        outcomes = shot_loop([first, second], 8)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0].keys(), ["a", "b"])
        self.assertEqual(outcomes[0]["a"].type, bit[8, 1])
        self.assertEqual(outcomes[1]["c"].type, bit[8, 2])

    def test_structure_navigates_like_a_container(self):
        """Test that a structure of values is indexed and iterated like what it describes."""
        first = self.circuit(registers=(("a", 1), ("b", 2)))
        outcomes = shot_loop([first, self.circuit()], 8)
        entry = outcomes[0]
        self.assertEqual([len(value) for value in outcomes], [2, 1])
        self.assertEqual([name for name, _ in entry.items()], ["a", "b"])
        self.assertEqual([value.type for value in entry.values()], [bit[8, 1], bit[8, 2]])
        self.assertEqual(outcomes[-1].keys(), ["c"])
        self.assertIn("b", entry)
        self.assertNotIn("z", entry)
        with self.assertRaises(KeyError):
            _ = entry["z"]
        with self.assertRaisesRegex(IndexError, "addresses nothing among 2 values"):
            _ = outcomes[2]
        with self.assertRaisesRegex(TypeError, "indexed by position"):
            _ = outcomes["a"]

    def test_one_value_has_no_structure(self):
        """Test that a single value raises on the container operations but is still a value."""
        bits = shot_loop([self.circuit()], 8)[0]["c"]
        for operation in (len, iter, list):
            with self.assertRaisesRegex(TypeError, r"a single value of type Bit\[8, 2\]"):
                operation(bits)
        self.assertTrue(bits)

    def test_a_structure_has_no_type_of_its_own(self):
        """Test that asking a structure of values for one type says to index into it."""
        outcomes = shot_loop([self.circuit()], 8)
        with self.assertRaisesRegex(TypeError, "index into it first"):
            _ = outcomes.type

    def test_every_value_shares_one_node(self):
        """Test that the values reached from one shot loop are positions in one node."""
        outcomes = shot_loop([self.circuit(registers=(("a", 1), ("b", 2)))], 8)
        program = build({"a": outcomes[0]["a"], "b": outcomes[0]["b"]})
        self.assertEqual(
            program._node_type_counts(),
            {"qiskit.constant": 1, "qiskit.shot_loop": 1, "qiskit.result": 2},
        )

    def test_outcomes_can_be_the_outputs(self):
        """Test that a structure of values is built as it is and comes back the same way."""
        outcomes = shot_loop([self.circuit(registers=(("a", 1), ("b", 2)))], 8)
        program = build(outcomes)
        self.assertEqual(program.output_types(), DataTree([{"a": bit[8, 1], "b": bit[8, 2]}]))
        self.assertEqual(build({"all": outcomes}).output_types()["all.0.b"], bit[8, 2])

    def test_each_value_is_its_own_register(self):
        """Test that the register indexed out of a shot loop is the one wired into the program."""
        first = self.circuit(registers=(("a", 1), ("b", 2), ("g", 3)))
        second = self.circuit(registers=(("d", 4), ("e", 5)))
        # The circuit in the middle measures nothing, so it contributes no value to index past.
        outcomes = shot_loop([first, self.circuit(registers=()), second], 8)
        program = build([outcomes[2]["e"], outcomes[0]["b"], outcomes[2]["d"], outcomes[0]["g"]])
        self.assertEqual(
            program.output_types(), DataTree([bit[8, 5], bit[8, 2], bit[8, 4], bit[8, 3]])
        )

    def test_a_structure_is_not_one_output(self):
        """Test that a structure held as one leaf is refused, since it is not one value."""
        outcomes = shot_loop([self.circuit()], 8)
        with self.assertRaisesRegex(TypeError, "structure of values"):
            build(DataTree.leaf_of(outcomes))

    def test_batch_axes_carry_onto_the_outcomes(self):
        """Test that any rank of leading axes on the parameter values reaches the outcomes."""
        circuit = self.circuit((Parameter("a"),))
        for shape, batch in ((f64[1], ()), (f64[4, 1], (4,)), (f64[2, 3, 1], (2, 3))):
            with self.subTest(shape):
                outcomes = shot_loop([circuit], 8, [qp_input("values", shape)])
                self.assertEqual(outcomes[0]["c"].type, bit[(*batch, 8, 2)])

    def test_values_of_the_wrong_type_raise_where_written(self):
        """Test that values that are not a circuit's parameters raise, naming the circuit."""
        circuit = self.circuit((Parameter("a"), Parameter("b")))
        with self.assertRaisesRegex(
            ValueError, r"circuit 0: expected .* \[\.\.\., 2\], got F64\[3\]"
        ):
            shot_loop([circuit], 8, [qp_input("values", f64[3])])

    def test_a_circuit_taking_no_parameters_needs_no_values(self):
        """Test that a circuit taking no parameters is wired in with nothing given for it."""
        angles = qp_input("angles", f64[4, 1])
        outcomes = shot_loop([self.circuit((Parameter("a"),)), self.circuit()], 8, [angles, None])
        self.assertEqual(outcomes[0]["c"].type, bit[4, 8, 2])
        self.assertEqual(outcomes[1]["c"].type, bit[8, 2], "no values means no batch")
        # Every circuit taking no parameters shares the one empty set of values.
        program = build(shot_loop([self.circuit(), self.circuit()], 8))
        self.assertEqual(program._node_type_counts()["qiskit.constant"], 1)

    def test_one_set_of_values_per_circuit(self):
        """Test that a shot loop needs one set of parameter values for each of its circuits."""
        with self.assertRaisesRegex(ValueError, "2 circuits, 1 given"):
            shot_loop([self.circuit(), self.circuit()], 8, [None])

    def test_a_circuit_taking_parameters_is_not_given_none(self):
        """Test that leaving out the values of a circuit that takes some raises, naming it."""
        with self.assertRaisesRegex(
            ValueError, r"circuit 0: expected .* \[\.\.\., 1\], got F64\[0\]"
        ):
            shot_loop([self.circuit((Parameter("a"),))], 8)

    def test_only_circuits_can_be_run(self):
        """Test that something that is not a circuit is refused, naming it and its type."""
        with self.assertRaisesRegex(TypeError, "circuit 1: expected a QuantumCircuit, got int"):
            shot_loop([self.circuit(), 1], 8)

    def test_a_circuit_that_builds_itself_is_built_first(self):
        """Test that a circuit assembled on demand is assembled before it is wired in."""

        class Lazy(QuantumCircuit):
            """A circuit that adds its register and its rotation when its data is read."""

            @property
            def data(self):
                if not self.cregs:
                    self.add_register(ClassicalRegister(2, "c"))
                    self.rx(Parameter("a"), 0)
                    self.measure([0, 1], [0, 1])
                return super().data

        outcomes = shot_loop([Lazy(2)], 8, [qp_input("values", f64[1])])
        self.assertEqual(outcomes[0]["c"].type, bit[8, 2])

    def test_expressions_supply_a_circuits_parameters(self):
        """Test that expressions evaluated in the program feed a circuit's parameter values."""
        angle = Parameter("angle")
        circuit = self.circuit((Parameter("a"), Parameter("b")))
        values = bind_parameters([angle, 2 * angle], qp_input("angles", f64[3, 1]), [angle])
        self.assertEqual(values.type, f64[3, 2])
        self.assertEqual(shot_loop([circuit], 8, [values])[0]["c"].type, bit[3, 8, 2])

    def test_surplus_declared_parameters_accepted(self):
        """Test that declaring a parameter no expression uses is accepted."""
        a, b = Parameter("a"), Parameter("b")
        values = bind_parameters([a], qp_input("x", f64[2]), [a, b])
        self.assertEqual(values.type, f64[1])

    def test_undeclared_parameter_named(self):
        """Test that an expression over a parameter that is not declared raises, naming it."""
        a, b = Parameter("a"), Parameter("b")
        with self.assertRaisesRegex(ValueError, "undeclared parameter b"):
            bind_parameters([a + b], qp_input("x", f64[1]), [a])

    def test_expressions_run_in_process(self):
        """Test that binding parameters is arithmetic the program can perform itself."""
        angle = Parameter("angle")
        program = build(bind_parameters([angle, 2 * angle], qp_input("x", f64[2, 1]), [angle]))
        np.testing.assert_allclose(program(x=[[0.5], [1.5]]).leaf, [[0.5, 1.0], [1.5, 3.0]])

    def test_a_program_of_circuits_needs_a_backend(self):
        """Test that calling a program holding circuits names the node it cannot perform."""
        values = qp_input("values", f64[1])
        program = build(shot_loop([self.circuit((Parameter("a"),))], 8, [values]))
        with self.assertRaisesRegex(ValueError, r"\(qiskit.shot_loop\) has no built-in"):
            program(values=[0.5])

    def test_circuits_parameters_and_post_processing_in_one_program(self):
        """Test that computing parameter values, running circuits and reducing them compose."""
        angle = Parameter("angle")
        circuit = self.circuit((Parameter("a"), Parameter("b")), (("meas", 2),))
        angles = qp_input("angles", f64[16, 1])
        values = bind_parameters([angle, 2 * angle], angles, [angle])
        bits = shot_loop([circuit], 1024, [values])[0]["meas"]
        program = build({"excited": bits.mean(axis=1), "parity": bits.parity(2).mean(1)})

        self.assertEqual(program.input_types(), DataTree({"angles": f64[16, 1]}))
        self.assertEqual(
            program.output_types(), DataTree({"excited": f64[16, 2], "parity": f64[16]})
        )
        self.assertEqual(
            program._node_type_counts(),
            {
                "qiskit.parameter": 1,
                "qiskit.bind_parameters": 1,
                "qiskit.shot_loop": 1,
                "qiskit.parity": 1,
                "qiskit.mean": 2,
                "qiskit.result": 2,
            },
        )
        with self.assertRaisesRegex(ValueError, "has no built-in implementation"):
            program(angles=np.linspace(0.0, np.pi, 16).reshape(16, 1))
