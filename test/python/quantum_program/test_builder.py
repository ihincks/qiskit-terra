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

"""Tests for the tracer front-end that builds a QuantumProgram."""

from collections import namedtuple

import numpy as np

from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, QuantumRegister
from qiskit.quantum_program import (
    DataTree,
    TensorSpec,
    Tracer,
    add,
    bit,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    build,
    constant,
    divide,
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
from test import QiskitTestCase


class TestTensorSpec(QiskitTestCase):
    """Tests for TensorSpec and the dtype sugar in qiskit.quantum_program.dtypes."""

    def test_dtype_sugar_fixed_shape(self):
        """Indexing a dtype with ints builds a TensorSpec with fixed dims."""
        spec = f64[3, 4]
        self.assertEqual(spec.dtype, "f64")
        self.assertEqual(spec.shape, [3, 4])

    def test_dtype_sugar_scalar(self):
        """Indexing a dtype with an empty tuple builds a scalar TensorSpec."""
        spec = f64[()]
        self.assertEqual(spec.shape, [])

    def test_dtype_sugar_named_dim(self):
        """Indexing a dtype with a string builds a named (symbolic) dim."""
        spec = f64["n"]
        self.assertEqual(spec.shape, ["n"])

    def test_tensor_spec_direct_construction(self):
        """TensorSpec can also be constructed directly with a dtype name and shape."""
        spec = TensorSpec("bit", [2, "n"])
        self.assertEqual(spec.dtype, "bit")
        self.assertEqual(spec.shape, [2, "n"])


class TestDataTree(QiskitTestCase):
    """Tests for DataTree, the Python mirror of the Rust data_tree used for structured
    Tracer outputs and the outer output structure passed to build()."""

    def test_round_trip_nested_containers(self):
        """from_python/to_python round-trips nested list/tuple/dict, preserving shape."""
        obj = {"a": [1, 2, (3, 4)], "b": {"c": 5}}
        tree = DataTree.from_python(obj)
        self.assertEqual(tree.to_python(), obj)

    def test_round_trip_preserves_namedtuple_type(self):
        """A namedtuple round-trips back to the same namedtuple type, not a plain tuple."""
        Point = namedtuple("Point", ["x", "y"])
        tree = DataTree.from_python(Point(1, 2))
        result = tree.to_python()
        self.assertIsInstance(result, Point)
        self.assertEqual(result, Point(1, 2))

    def test_dict_key_order_preserved_not_sorted(self):
        """A dict's insertion order is preserved, not sorted."""
        tree = DataTree.from_python({"z": 1, "a": 2, "m": 3})
        self.assertEqual(tree.keys, ("z", "a", "m"))

    def test_paths_and_dotted_paths(self):
        """paths()/dotted_paths() walk leaves in depth-first order."""
        tree = DataTree.from_python({"a": [1, 2], "b": 3})
        self.assertEqual(tree.paths(), [["a", 0], ["a", 1], ["b"]])
        self.assertEqual(tree.dotted_paths(), ["a.0", "a.1", "b"])

    def test_bare_leaf_path_is_out(self):
        """A tree with no branch at all is a single leaf, dotted-path "out"."""
        tree = DataTree.from_python(42)
        self.assertEqual(tree.paths(), [[]])
        self.assertEqual(tree.dotted_paths(), ["out"])

    def test_num_leaves(self):
        """num_leaves() counts every leaf in the tree."""
        tree = DataTree.from_python([1, {"a": 2, "b": 3}, (4,)])
        self.assertEqual(tree.num_leaves(), 4)

    def test_unflatten_arity_mismatch_raises(self):
        """unflatten() raises ValueError if given the wrong number of values."""
        tree = DataTree.from_python([1, 2, 3])
        with self.assertRaises(ValueError):
            tree.unflatten([1, 2])

    def test_unflatten_round_trip(self):
        """unflatten() rebuilds a same-shaped tree from new leaf values."""
        tree = DataTree.from_python({"a": [1, 2], "b": 3})
        rebuilt = tree.unflatten([10, 20, 30])
        self.assertEqual(rebuilt.to_python(), {"a": [10, 20], "b": 30})

    def test_get_by_path(self):
        """get_by_path() walks a mix of indices and keys, or returns None if invalid."""
        tree = DataTree.from_python({"a": [1, 2], "b": 3})
        self.assertEqual(tree.get_by_path(["a", 1]).value, 2)
        self.assertIsNone(tree.get_by_path(["z"]))
        self.assertIsNone(tree.get_by_path(["a", 5]))
        self.assertIsNone(tree.get_by_path(["b", 0]))  # descending through a leaf


class TestTracer(QiskitTestCase):
    """Tests for the tracer front-end: eager type inference and build()."""

    def test_input_returns_tracer(self):
        """qp_input declares an input and returns a matching tracer."""
        x = qp_input("x", f64[3])
        self.assertIsInstance(x, Tracer)
        self.assertEqual(x.dtype, "f64")
        self.assertEqual(x.shape, [3])

    def test_fanout_single_input_one_multiply(self):
        """x * x shares one input and one multiply node after build (fan-out)."""
        x = qp_input("x", f64[3])
        z = x * x
        prog = build({"z": z})
        self.assertEqual(prog._node_labels(), ["__input_x", "multiply_0"])
        self.assertEqual(prog.input_keys(), ["x"])
        self.assertEqual(prog.resolve(), {"z": TensorSpec("f64", [3])})

    def test_mean_reduction(self):
        """mean(axis=...) removes the reduced axis and produces a scalar here."""
        x = qp_input("x", f64[3])
        z = (x * x).mean(axis=0)
        # The spec is available eagerly, before any build().
        self.assertEqual(z.spec, TensorSpec("f64", []))
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [])})

    def test_var_and_std_reduction(self):
        """var(axis=...) and std(axis=...) both remove the reduced axis."""
        x = qp_input("x", f64[3, 4])
        v = x.var(axis=0, ddof=1.0)
        s = x.std(axis=1)
        resolved = build({"v": v, "s": s}).resolve()
        self.assertEqual(resolved["v"], TensorSpec("f64", [4]))
        self.assertEqual(resolved["s"], TensorSpec("f64", [3]))

    def test_parity_reduction(self):
        """parity(axis=...) reduces a bit tensor via XOR, removing that axis."""
        x = qp_input("x", bit[4])
        z = x.parity(axis=0)
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("bit", [])})

    def test_shape_mismatch_raises_at_op_line(self):
        """Incompatible shapes raise ValueError immediately during tracing."""
        x = qp_input("x", f64[3])
        y = qp_input("y", f64[4])
        with self.assertRaises(ValueError):
            x + y  # pylint: disable=pointless-statement

    def test_failed_op_does_not_block_later_build(self):
        """A failed op raises during tracing; a separate valid build still works."""
        x = qp_input("x", f64[3])
        y = qp_input("y", f64[4])
        with self.assertRaises(ValueError):
            x + y  # pylint: disable=pointless-statement
        # Tracing is pure; the failed attempt leaves no residue, and a fresh build works.
        prog = build({"x": x, "y": y})
        self.assertEqual(
            prog.resolve(),
            {"x": TensorSpec("f64", [3]), "y": TensorSpec("f64", [4])},
        )

    def test_constant_and_arithmetic(self):
        """A numpy array can be wired in as a constant and combined with a tracer."""
        x = qp_input("x", f64[3])
        c = constant(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(c.dtype, "f64")
        self.assertEqual(c.shape, [3])
        z = x + c
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [3])})

    def test_python_scalar_coerced_to_constant(self):
        """A bare Python/numpy scalar operand is coerced into a constant automatically."""
        x = qp_input("x", f64[3])
        z = x + 1.0
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [3])})

    def test_reflected_operator(self):
        """Reflected dunders (e.g. __radd__) work when the tracer is on the right."""
        x = qp_input("x", f64[3])
        z = 1.0 + x
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [3])})

    def test_bitwise_ops(self):
        """Bitwise and/or/xor/not are exposed via the usual Python operators."""
        x = qp_input("x", bit[4])
        y = qp_input("y", bit[4])
        prog = build({"and": x & y, "or": x | y, "xor": x ^ y, "not": ~x})
        for spec in prog.resolve().values():
            self.assertEqual(spec, TensorSpec("bit", [4]))

    def test_power(self):
        """__pow__ builds a power node; three-argument pow() is rejected."""
        x = qp_input("x", f64[3])
        z = x**2
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [3])})
        with self.assertRaises(ValueError):
            pow(x, 2, 5)

    def test_output_keys(self):
        """output_keys reflects every declared output, in declaration order."""
        x = qp_input("x", f64[3])
        prog = build({"a": x, "b": x * x})
        self.assertEqual(prog.output_keys(), ["a", "b"])

    def test_dtype_mismatch_promotes_rather_than_errors(self):
        """Mixed-dtype elementwise ops promote rather than raising."""
        x = qp_input("x", f64[3])
        y = qp_input("y", i64[3])
        z = x + y
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [3])})

    def test_dce_unreferenced_op_not_materialized(self):
        """Tracers not reachable from the requested outputs are never materialized."""
        x = qp_input("x", f64[3])
        _dead = x * x  # never referenced by an output
        live = x + 1.0
        prog = build({"live": live})
        # Only the input and the "add" survive; the unreferenced multiply is dropped.
        self.assertEqual(prog._node_labels(), ["__input_x", "add_0", "constant_0"])
        self.assertEqual(prog.resolve(), {"live": TensorSpec("f64", [3])})

    def test_build_output_pytree_forms(self):
        """build() accepts a single tracer, a list, a tuple, or a dict of outputs."""
        x = qp_input("x", f64[3])

        single = build(x)
        self.assertEqual(single.output_keys(), ["out"])

        as_list = build([x, x * x])
        self.assertEqual(as_list.output_keys(), ["0", "1"])

        as_tuple = build((x, x + 1.0))
        self.assertEqual(as_tuple.output_keys(), ["0", "1"])

        as_dict = build({"a": x, "b": x * x})
        self.assertEqual(as_dict.output_keys(), ["a", "b"])

    def test_duplicate_input_key_conflicting_spec_errors(self):
        """Two inputs with the same key but different specs raise at build time."""
        x1 = qp_input("x", f64[3])
        x2 = qp_input("x", f64[4])
        with self.assertRaises(ValueError):
            build({"a": x1, "b": x2})

    def test_duplicate_input_key_matching_spec_dedups(self):
        """Two inputs with the same key and matching spec collapse to one input node."""
        x1 = qp_input("x", f64[3])
        x2 = qp_input("x", f64[3])
        prog = build({"a": x1 + 1.0, "b": x2 * 2.0})
        self.assertEqual(prog.input_keys(), ["x"])

    def test_nested_output_structure_round_trips(self):
        """A nested list/dict of outputs round-trips through build().resolve() unchanged."""
        x = qp_input("x", f64[3])
        prog = build({"a": [x, {"b": x * x}]})
        self.assertEqual(prog.output_keys(), ["a.0", "a.1.b"])
        self.assertEqual(
            prog.resolve(), {"a": [TensorSpec("f64", [3]), {"b": TensorSpec("f64", [3])}]}
        )

    def test_build_non_tracer_leaf_raises_type_error(self):
        """A non-Tracer leaf in the outputs raises TypeError naming its path."""
        x = qp_input("x", f64[3])
        with self.assertRaises(TypeError):
            build({"a": x, "b": 5})

    def test_build_colliding_dotted_keys_raises_value_error(self):
        """Two outputs that flatten to the same dotted key raise ValueError."""
        x = qp_input("x", f64[3])
        with self.assertRaises(ValueError):
            build({"a.b": x, "a": {"b": x}})


class TestStandaloneOps(QiskitTestCase):
    """The numpy-style standalone functions build the identical node as their
    operator/method counterpart on Tracer."""

    def test_arithmetic_functions_match_operators(self):
        """add/subtract/multiply/divide/remainder/power match their operator counterparts."""
        x = qp_input("x", f64[3])
        y = qp_input("y", f64[3])
        pairs = [
            (add(x, y), x + y),
            (subtract(x, y), x - y),
            (multiply(x, y), x * y),
            (divide(x, y), x / y),
            (remainder(x, y), x % y),
            (power(x, y), x**y),
        ]
        for from_function, from_operator in pairs:
            self.assertEqual(from_function._node.op, from_operator._node.op)
            self.assertEqual(from_function.spec, from_operator.spec)

    def test_bitwise_functions_match_operators(self):
        """bitwise_and/or/xor/not match their operator counterparts."""
        x = qp_input("x", bit[4])
        y = qp_input("y", bit[4])
        pairs = [
            (bitwise_and(x, y), x & y),
            (bitwise_or(x, y), x | y),
            (bitwise_xor(x, y), x ^ y),
            (bitwise_not(x), ~x),
        ]
        for from_function, from_operator in pairs:
            self.assertEqual(from_function._node.op, from_operator._node.op)
            self.assertEqual(from_function.spec, from_operator.spec)

    def test_reduction_functions_match_methods(self):
        """mean/var/std/parity match their Tracer-method counterparts."""
        x = qp_input("x", f64[3, 4])
        b = qp_input("b", bit[4])
        pairs = [
            (mean(x, axis=0), x.mean(axis=0)),
            (var(x, axis=0, ddof=1.0), x.var(axis=0, ddof=1.0)),
            (std(x, axis=1), x.std(axis=1)),
            (parity(b, axis=0), b.parity(axis=0)),
        ]
        for from_function, from_method in pairs:
            self.assertEqual(from_function._node.op, from_method._node.op)
            self.assertEqual(from_function.spec, from_method.spec)

    def test_standalone_functions_resolve_end_to_end(self):
        """A pipeline built entirely from standalone functions resolves like its operator form."""
        x = qp_input("x", f64[3])
        z = mean(multiply(x, x), axis=0)
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [])})

    def test_standalone_functions_coerce_python_scalars(self):
        """Standalone binary functions coerce a bare scalar operand, like the operators do."""
        x = qp_input("x", f64[3])
        z = add(x, 1.0)
        self.assertEqual(build({"z": z}).resolve(), {"z": TensorSpec("f64", [3])})


class TestPortModel(QiskitTestCase):
    """Tests for the Tracer-as-port model: structure, indexing, and identity."""

    def test_structured_tracer_rejects_spec_dtype_shape(self):
        """.spec/.dtype/.shape raise TypeError on a structured (non-leaf) tracer."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()
        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        for attr in ("spec", "dtype", "shape"):
            with self.assertRaises(TypeError):
                getattr(outputs, attr)  # pylint: disable=expression-not-assigned

    def test_structured_tracer_rejects_arithmetic(self):
        """Arithmetic operators raise TypeError on a structured (non-leaf) tracer."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()
        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        with self.assertRaises(TypeError):
            outputs + 1  # pylint: disable=pointless-statement

    def test_structured_tracer_bad_index_raises(self):
        """Indexing a structured tracer with a bad key/index raises KeyError/IndexError."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()
        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        with self.assertRaises(IndexError):
            outputs[5]  # pylint: disable=pointless-statement
        with self.assertRaises(KeyError):
            outputs[0]["nonexistent"]  # pylint: disable=pointless-statement

    def test_leaf_tracer_rejects_indexing_and_iteration(self):
        """Indexing or iterating a leaf tracer raises TypeError."""
        x = qp_input("x", f64[3])
        with self.assertRaises(TypeError):
            x[0]  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            iter(x)

    def test_repeated_indexing_shares_one_node(self):
        """Fetching x[0]["meas"] twice materializes to a single shared node."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()
        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        prog = build({"a": outputs[0]["meas"], "b": outputs[0]["meas"]})
        self.assertEqual(
            [label for label in prog._node_labels() if label.startswith("shot_loop")],
            ["shot_loop_0"],
        )


class TestShotLoop(QiskitTestCase):
    """Tests for shot_loop, which wires in a ShotLoop node via the tracer front-end."""

    def test_single_circuit_pipeline_resolves(self):
        """A shot_loop output can be fed through parity/mean/affine to a scalar output."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.ry(theta, 1)
        qc.measure_all()

        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=4000, params=[theta_in])
        self.assertEqual(len(outputs), 1)
        bits = outputs[0]["meas"]
        self.assertIsInstance(bits, Tracer)
        self.assertEqual(bits.dtype, "bit")
        self.assertEqual(bits.shape, [4000, 2])

        magnetization = 1 - 2 * bits.parity(axis=1).mean(axis=0)
        prog = build({"magnetization": magnetization})
        self.assertEqual(prog.resolve(), {"magnetization": TensorSpec("f64", [])})

    def test_multiple_circuits_with_different_registers(self):
        """Each circuit gets its own dict of tracers, keyed by register name."""
        theta = Parameter("theta")

        qc0 = QuantumCircuit(2, name="qc0")
        qc0.ry(theta, [0, 1])
        qc0.measure_all()  # creates a single "meas" register of length 2

        qc1 = QuantumCircuit(
            QuantumRegister(3), ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
        )
        qc1.rx(theta, [0, 1, 2])
        qc1.measure(0, 0)
        qc1.measure([1, 2], [1, 2])

        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc0, qc1], shots=100, params=[theta_in, theta_in])

        self.assertEqual(len(outputs), 2)
        self.assertEqual(set(outputs[0]), {"meas"})
        self.assertEqual(outputs[0]["meas"].dtype, "bit")
        self.assertEqual(outputs[0]["meas"].shape, [100, 2])
        self.assertEqual(set(outputs[1]), {"a", "b"})
        self.assertEqual(outputs[1]["a"].shape, [100, 1])
        self.assertEqual(outputs[1]["b"].shape, [100, 2])

    def test_shot_loop_single_node_shared_across_projections(self):
        """All per-register projections resolve to one shared shot_loop node at build."""
        theta = Parameter("theta")
        qc = QuantumCircuit(
            QuantumRegister(3), ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
        )
        qc.rx(theta, [0, 1, 2])
        qc.measure(0, 0)
        qc.measure([1, 2], [1, 2])

        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        prog = build({"a": outputs[0]["a"], "b": outputs[0]["b"]})
        # Exactly one shot_loop node backs both register projections.
        self.assertEqual(
            [label for label in prog._node_labels() if label.startswith("shot_loop")],
            ["shot_loop_0"],
        )

    def test_params_length_mismatch_raises(self):
        """len(params) must equal len(circuits)."""
        qc = QuantumCircuit(1)
        qc.measure_all()
        theta_in = qp_input("theta", f64[1])
        with self.assertRaises(ValueError):
            shot_loop([qc], shots=10, params=[])
        with self.assertRaises(ValueError):
            shot_loop([qc], shots=10, params=[theta_in, theta_in])

    def test_wrong_param_shape_raises_at_shot_loop(self):
        """A params tracer with the wrong dtype/shape raises ValueError during tracing."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()

        wrong_shape = qp_input("wrong_shape", f64[2])
        wrong_dtype = qp_input("wrong_dtype", bit[1])
        with self.assertRaises(ValueError):
            shot_loop([qc], shots=10, params=[wrong_shape])
        with self.assertRaises(ValueError):
            shot_loop([qc], shots=10, params=[wrong_dtype])

    def test_single_batch_dim_prepended_to_output(self):
        """A single leading batch dim on params is prepended to each output register shape."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.ry(theta, 1)
        qc.measure_all()

        theta_in = qp_input("theta", f64[10, 1])
        outputs = shot_loop([qc], shots=4000, params=[theta_in])
        self.assertEqual(outputs[0]["meas"].shape, [10, 4000, 2])

    def test_multiple_batch_dims_prepended_to_output(self):
        """Multiple leading batch dims on params are all prepended to the output shape."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.ry(theta, 1)
        qc.measure_all()

        theta_in = qp_input("theta", f64[10, 20, 30, 1])
        outputs = shot_loop([qc], shots=4000, params=[theta_in])
        self.assertEqual(outputs[0]["meas"].shape, [10, 20, 30, 4000, 2])

    def test_batched_pipeline_resolves(self):
        """A batched shot_loop output can be reduced downstream, accounting for the extra
        leading batch axes when choosing which axis to reduce over."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.ry(theta, 1)
        qc.measure_all()

        theta_in = qp_input("theta", f64[10, 1])
        outputs = shot_loop([qc], shots=4000, params=[theta_in])
        bits = outputs[0]["meas"]
        self.assertEqual(bits.shape, [10, 4000, 2])

        # bits has shape [10, 4000, 2]; the "shots" axis is now 1, not 0, because of the
        # leading batch dim.
        magnetization = 1 - 2 * bits.parity(axis=2).mean(axis=1)
        prog = build({"magnetization": magnetization})
        self.assertEqual(prog.resolve(), {"magnetization": TensorSpec("f64", [10])})

    def test_wrong_trailing_dim_with_batch_prefix_raises(self):
        """A wrong trailing (parameter-count) dim still raises even with a batch prefix
        present."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()

        wrong_shape = qp_input("wrong_shape", f64[10, 2])
        with self.assertRaises(ValueError):
            shot_loop([qc], shots=10, params=[wrong_shape])

    def test_build_directly_on_shot_loop_result_round_trips(self):
        """build() accepts the raw shot_loop() result directly, no wrapping dict needed, and
        resolve() gives back the same list[dict[str, TensorSpec]] shape."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()
        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        prog = build(outputs)
        self.assertEqual(prog.resolve(), [{"meas": TensorSpec("bit", [10, 1])}])

    def test_build_directly_on_one_circuit_result_round_trips(self):
        """build() on a single circuit's dict of registers resolves back to that dict."""
        qc = QuantumCircuit(
            QuantumRegister(3), ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
        )
        theta = Parameter("theta")
        qc.rx(theta, [0, 1, 2])
        qc.measure(0, 0)
        qc.measure([1, 2], [1, 2])

        theta_in = qp_input("theta", f64[1])
        outputs = shot_loop([qc], shots=10, params=[theta_in])
        prog = build(outputs[0])
        self.assertEqual(
            prog.resolve(), {"a": TensorSpec("bit", [10, 1]), "b": TensorSpec("bit", [10, 2])}
        )
