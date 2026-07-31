// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! PyO3 bindings backing the Python-side `qiskit.quantum_program` tracer surface.
//!
//! The ergonomic Python API is a *tracer* (`qiskit/quantum_program/_tracer.py`): operator
//! calls build pure-Python expression objects and no graph exists during construction. A
//! real [`QuantumProgram`] graph is only materialized when the user calls `build(...)`.
//! This module supplies the two halves that engine needs:
//!
//! * **Eager abstract-eval** — free `#[pyfunction]`s ([`infer_op`], [`infer_shot_loop`],
//!   [`spec_of_array`]) compute a [`PyTensorSpec`] for an operation from its input specs by
//!   constructing the concrete node and delegating to its `resolve_types_flat`, with *no*
//!   graph. This keeps the dtype/shape rules single-sourced in Rust and surfaces errors at
//!   the offending Python line.
//! * **Graph materialization** — the underscore-prefixed builder methods on
//!   [`PyQuantumProgram`] (`_add_op`, `_add_input`, `_add_constant`, `_add_shot_loop`,
//!   `_add_edge`, `_set_output`, ...) drive the incremental Rust builder. Node labeling
//!   and expression-graph traversal live in Python.
//!
//! The op → concrete-node dispatch is centralized in the [`with_math_node`] macro so the
//! "add a node" path and the "infer types" path cannot drift apart. This is required
//! because `DynNode`/`DynNode::new` are private: the only type-erasure path is the generic
//! [`QuantumProgram::add_node`], so each op must be a `match` arm that names its own
//! concrete node type.

use crate::data_tree::DataTree;
use crate::math_nodes::{
    Add, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Divide, Mean, Multiply, Parity, Power,
    Remainder, Std, Subtract, Variance,
};
use crate::parameter_expressions::ParameterExpressions;
use crate::program_node::ProgramNode;
use crate::quantum_program::{OwnedPath, OwnedPathEntry, Port, QuantumProgram};
use crate::shot_loop::ShotLoop;
use crate::store::Store;
use crate::tensor::{DType, DTypeLike, Dim, Tensor, TensorType};

use numpy::{Complex32, Complex64, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::parameter::parameter_expression::PyParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;

/// Map any `Display`-able error into a Python `ValueError`.
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse a `TensorSpec`/dtype-sugar dtype name (e.g. `"f64"`) into a [`DType`].
fn parse_dtype(name: &str) -> PyResult<DType> {
    match name {
        "c128" => Ok(DType::C128),
        "c64" => Ok(DType::C64),
        "f64" => Ok(DType::F64),
        "f32" => Ok(DType::F32),
        "i64" => Ok(DType::I64),
        "i32" => Ok(DType::I32),
        "i16" => Ok(DType::I16),
        "i8" => Ok(DType::I8),
        "u64" => Ok(DType::U64),
        "u32" => Ok(DType::U32),
        "u16" => Ok(DType::U16),
        "u8" => Ok(DType::U8),
        "bit" => Ok(DType::Bit),
        other => Err(PyValueError::new_err(format!("unknown dtype {other:?}"))),
    }
}

/// Format a shape as e.g. `[3, "n"]`, for `__repr__` implementations.
fn format_shape(shape: &[Dim]) -> String {
    let entries: Vec<String> = shape
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => n.to_string(),
            Dim::Named(s) => format!("{s:?}"),
        })
        .collect();
    format!("[{}]", entries.join(", "))
}

/// Convert a resolved (necessarily concrete-dtype) [`TensorType`] into a [`PyTensorSpec`].
///
/// By construction, every `TensorType` flowing through this module's inference helpers has
/// a concrete dtype (program inputs and constants are always declared with one, and dtype
/// promotion of concrete dtypes stays concrete) — so a non-concrete dtype here indicates an
/// internal invariant violation, not a user error.
fn tensor_type_to_spec(ty: &TensorType) -> PyResult<PyTensorSpec> {
    let DTypeLike::Concrete(dtype) = ty.dtype else {
        return Err(PyValueError::new_err(
            "internal error: expected a fully-resolved concrete dtype",
        ));
    };
    Ok(PyTensorSpec {
        dtype,
        shape: ty.shape.clone(),
    })
}

/// Convert a numpy array-like Python object into a [`Tensor`], inferring the dtype from
/// the array's own numpy dtype. Tries each supported element type in turn; numpy's
/// downcast check requires an exact dtype match, so at most one attempt can succeed.
fn numpy_to_tensor(arr: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    macro_rules! try_dtype {
        ($t:ty, $variant:ident) => {
            if let Ok(a) = arr.extract::<PyReadonlyArrayDyn<$t>>() {
                return Ok(Tensor::$variant(a.as_array().to_owned().into_shared()));
            }
        };
    }
    try_dtype!(f64, F64);
    try_dtype!(f32, F32);
    try_dtype!(i64, I64);
    try_dtype!(i32, I32);
    try_dtype!(i16, I16);
    try_dtype!(i8, I8);
    try_dtype!(u64, U64);
    try_dtype!(u32, U32);
    try_dtype!(u16, U16);
    try_dtype!(u8, U8);
    try_dtype!(Complex64, C128);
    try_dtype!(Complex32, C64);
    if let Ok(a) = arr.extract::<PyReadonlyArrayDyn<bool>>() {
        let bits = a.as_array().mapv(|b| b as u8).into_shared();
        return Ok(Tensor::Bit(bits));
    }
    Err(PyValueError::new_err(
        "unsupported numpy array dtype for QuantumProgram constant",
    ))
}

/// Require and unwrap the `axis` argument of a reduction op, erroring if absent.
fn require_axis(op: &str, axis: Option<usize>) -> PyResult<usize> {
    axis.ok_or_else(|| PyValueError::new_err(format!("math op {op:?} requires an `axis` argument")))
}

/// Dispatch a math-op name to its concrete [`ProgramNode`], binding it to `$node` and
/// evaluating `$body` with that binding.
///
/// This is the single source of truth for the op → node mapping. Because each arm names a
/// distinct concrete type, `$body` is monomorphized per arm — which is exactly what both
/// call sites need: [`resolve_math_op`] calls `node.resolve_types_flat(..)` (the "infer"
/// path) and [`PyQuantumProgram::_add_op`] calls `self.inner.add_node(label, node)` (the
/// "add" path). Keeping both behind one macro guarantees they can't drift.
macro_rules! with_math_node {
    ($op:expr, $axis:expr, $ddof:expr, |$node:ident| $body:expr) => {{
        match $op {
            "add" => {
                let $node = Add;
                $body
            }
            "subtract" => {
                let $node = Subtract;
                $body
            }
            "multiply" => {
                let $node = Multiply;
                $body
            }
            "divide" => {
                let $node = Divide;
                $body
            }
            "remainder" => {
                let $node = Remainder;
                $body
            }
            "power" => {
                let $node = Power;
                $body
            }
            "bitwise_and" => {
                let $node = BitwiseAnd;
                $body
            }
            "bitwise_or" => {
                let $node = BitwiseOr;
                $body
            }
            "bitwise_xor" => {
                let $node = BitwiseXor;
                $body
            }
            "bitwise_not" => {
                let $node = BitwiseNot;
                $body
            }
            "mean" => {
                let $node = Mean::new(require_axis($op, $axis)?);
                $body
            }
            "variance" => {
                let $node = Variance::new(require_axis($op, $axis)?, $ddof.unwrap_or(0.0));
                $body
            }
            "std" => {
                let $node = Std::new(require_axis($op, $axis)?, $ddof.unwrap_or(0.0));
                $body
            }
            "parity" => {
                let $node = Parity::new(require_axis($op, $axis)?);
                $body
            }
            other => {
                return Err(PyValueError::new_err(format!("unknown math op {other:?}")));
            }
        }
    }};
}

/// Abstract-eval a math op: build its concrete node and run `resolve_types_flat` with no
/// graph. Shared by [`infer_op`] and (via the dispatch macro) the graph builder.
fn resolve_math_op(
    op: &str,
    input_types: &[TensorType],
    axis: Option<usize>,
    ddof: Option<f64>,
) -> PyResult<Vec<TensorType>> {
    with_math_node!(op, axis, ddof, |node| node
        .resolve_types_flat(input_types)
        .map_err(to_py_err))
}

/// Parse a Python path (list of `int`/`str`) into an [`OwnedPath`].
fn parse_path(entries: &[Bound<'_, PyAny>]) -> PyResult<OwnedPath> {
    entries
        .iter()
        .map(|entry| {
            if let Ok(i) = entry.extract::<usize>() {
                Ok(OwnedPathEntry::Index(i))
            } else if let Ok(s) = entry.extract::<String>() {
                Ok(OwnedPathEntry::Key(s))
            } else {
                Err(PyValueError::new_err("path entries must be int or str"))
            }
        })
        .collect()
}

/// A declared dtype + shape, used to declare program inputs and to carry the eagerly
/// computed type of each tracer.
///
/// Shape entries are either a fixed size (`int`) or a named, unresolved dimension
/// (`str`); named dimensions are passed through the type-inference engine untouched.
#[pyclass(
    module = "qiskit._accelerate.quantum_program",
    name = "TensorSpec",
    skip_from_py_object,
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyTensorSpec {
    dtype: DType,
    shape: Vec<Dim>,
}

#[pymethods]
impl PyTensorSpec {
    #[new]
    fn new(dtype: &str, shape: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        let shape = shape
            .iter()
            .map(|entry| {
                if let Ok(n) = entry.extract::<usize>() {
                    Ok(Dim::Fixed(n))
                } else if let Ok(s) = entry.extract::<String>() {
                    Ok(Dim::Named(s))
                } else {
                    Err(PyValueError::new_err("shape entries must be int or str"))
                }
            })
            .collect::<PyResult<Vec<Dim>>>()?;
        Ok(Self { dtype, shape })
    }

    #[getter]
    fn dtype(&self) -> String {
        self.dtype.to_string().to_lowercase()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for dim in &self.shape {
            match dim {
                Dim::Fixed(n) => list.append(*n)?,
                Dim::Named(s) => list.append(s.as_str())?,
            }
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorSpec({}, {})",
            self.dtype(),
            format_shape(&self.shape)
        )
    }
}

impl PyTensorSpec {
    fn to_tensor_type(&self) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(self.dtype),
            shape: self.shape.clone(),
            broadcastable: true,
        }
    }
}

/// Abstract-eval a math op from its input [`PyTensorSpec`]s, returning one output spec per
/// output leaf. Errors (shape/dtype mismatch, unknown op, missing axis) surface as
/// `ValueError`, letting the Python tracer type-check each operation at construction time.
#[pyfunction]
#[pyo3(signature = (op, specs, *, axis=None, ddof=None))]
fn infer_op(
    op: &str,
    specs: Vec<Bound<'_, PyTensorSpec>>,
    axis: Option<usize>,
    ddof: Option<f64>,
) -> PyResult<Vec<PyTensorSpec>> {
    let input_types: Vec<TensorType> = specs.iter().map(|s| s.borrow().to_tensor_type()).collect();
    let resolved = resolve_math_op(op, &input_types, axis, ddof)?;
    resolved.iter().map(tensor_type_to_spec).collect()
}

/// Abstract-eval a `ShotLoop` from its circuits and per-circuit parameter specs.
///
/// Returns one `dict[register_name -> TensorSpec]` per circuit, mirroring the shape of the
/// node's output tree, with no graph constructed. `param_specs[i]` supplies the parameter
/// values for `circuits[i]`.
#[pyfunction]
fn infer_shot_loop<'py>(
    py: Python<'py>,
    circuits: Vec<QuantumCircuitData<'py>>,
    shots: usize,
    param_specs: Vec<Bound<'py, PyTensorSpec>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    if circuits.len() != param_specs.len() {
        return Err(PyValueError::new_err(format!(
            "shot_loop got {} circuit(s) but {} params spec(s); these must match",
            circuits.len(),
            param_specs.len()
        )));
    }

    let circuit_data: Vec<_> = circuits.into_iter().map(|c| c.data).collect();
    let node = ShotLoop::new(circuit_data, shots);

    // Captured before `resolve_types_flat` so we can reshape the flat leaf list into one
    // dict per circuit, keyed by register name.
    let cregs: Vec<Vec<String>> = node
        .circuits()
        .iter()
        .map(|c| c.cregs().iter().map(|r| r.name().to_string()).collect())
        .collect();

    let param_types: Vec<TensorType> = param_specs
        .iter()
        .map(|s| s.borrow().to_tensor_type())
        .collect();
    let mut resolved = node
        .resolve_types_flat(&param_types)
        .map_err(to_py_err)?
        .into_iter();

    cregs
        .into_iter()
        .map(|regs| {
            let dict = PyDict::new(py);
            for name in regs {
                // Guaranteed present: `resolve_types_flat` yields exactly one leaf per
                // register, in the same per-circuit, per-register order as `cregs`.
                let ty = resolved
                    .next()
                    .expect("resolve_types_flat leaf count must match total register count");
                dict.set_item(name, tensor_type_to_spec(&ty)?)?;
            }
            Ok(dict)
        })
        .collect()
}

/// Build a [`ParameterExpressions`] node evaluating `expressions` from input values given for
/// `parameters`, in that order.
///
/// This is the single source of truth for turning Python-space arguments into the node, shared
/// by [`infer_parameter_expressions`] (the abstract-eval path) and
/// [`PyQuantumProgram::_add_parameter_expressions`] (the graph-building path), so the two can't
/// drift — the same role the [`with_math_node`] macro plays for the math ops.
fn make_parameter_expressions(
    expressions: Vec<PyParameterExpression>,
    parameters: Vec<Symbol>,
) -> PyResult<ParameterExpressions> {
    ParameterExpressions::new(
        expressions.into_iter().map(|p| p.inner).collect(),
        parameters,
    )
    .map_err(to_py_err)
}

/// Abstract-eval a `ParameterExpressions` node from its input spec (an `[..., N]`-shaped
/// batch of values for the `N` declared `parameters`). Returns the single resolved output spec
/// (`[..., M]`, `M` the number of expressions), with no graph constructed.
#[pyfunction]
fn infer_parameter_expressions(
    expressions: Vec<PyParameterExpression>,
    parameters: Vec<Symbol>,
    spec: &Bound<'_, PyTensorSpec>,
) -> PyResult<PyTensorSpec> {
    let resolved = make_parameter_expressions(expressions, parameters)?
        .resolve_types_flat(&[spec.borrow().to_tensor_type()])
        .map_err(to_py_err)?;
    tensor_type_to_spec(&resolved[0])
}

/// Infer the [`PyTensorSpec`] of a numpy array-like value (as wired in by `constant`).
///
/// Single-sources the dtype/shape inference used for constants: `np.asarray` → [`Tensor`]
/// → [`Tensor::tensor_type`] → spec.
#[pyfunction]
fn spec_of_array(value: &Bound<'_, PyAny>) -> PyResult<PyTensorSpec> {
    let py = value.py();
    let array = py.import("numpy")?.call_method1("asarray", (value,))?;
    let tensor = numpy_to_tensor(&array)?;
    tensor_type_to_spec(&tensor.tensor_type())
}

/// A data-flow graph of [`ProgramNode`]s, materialized from a Python tracer expression.
///
/// This is a thin wrapper around [`QuantumProgram`]. Node labeling and expression traversal
/// happen in Python (`_tracer.py`); the underscore-prefixed methods here are the low-level
/// builder the materializer drives.
///
/// `unsendable`: the underlying graph stores nodes as `Box<dyn ProgramNode>` trait objects,
/// which are not `Send`; the object is confined to the thread that created it (matching
/// CPython's default single-threaded access to any given object anyway).
#[pyclass(
    module = "qiskit._accelerate.quantum_program",
    name = "QuantumProgram",
    unsendable
)]
pub struct PyQuantumProgram {
    inner: QuantumProgram,
}

#[pymethods]
impl PyQuantumProgram {
    #[new]
    fn new() -> Self {
        Self {
            inner: QuantumProgram::new(),
        }
    }

    /// Declare a program input under `key` with the given [`PyTensorSpec`], returning the
    /// label of the created `Input` source node (whose single output port is at the root
    /// path). Fan-out is achieved by wiring that output to several consumers via `_add_edge`.
    fn _add_input(&mut self, key: &str, spec: &Bound<'_, PyTensorSpec>) -> PyResult<String> {
        let ty = spec.borrow().to_tensor_type();
        let port = self.inner.add_input(key, ty).map_err(to_py_err)?;
        Ok(port.label)
    }

    /// Add a constant (`Store`) node under `label` holding `value` (coerced via `np.asarray`).
    fn _add_constant(&mut self, label: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let array = py.import("numpy")?.call_method1("asarray", (value,))?;
        let tensor = numpy_to_tensor(&array)?;
        self.inner
            .add_node(label, Store::new(DataTree::new_leaf(tensor)))
            .map_err(to_py_err)
    }

    /// Add a math-op node under `label`. `axis`/`ddof` apply to reduction ops only.
    #[pyo3(signature = (label, op, axis=None, ddof=None))]
    fn _add_op(
        &mut self,
        label: &str,
        op: &str,
        axis: Option<usize>,
        ddof: Option<f64>,
    ) -> PyResult<()> {
        with_math_node!(op, axis, ddof, |node| self
            .inner
            .add_node(label, node)
            .map_err(to_py_err))
    }

    /// Add a `ShotLoop` node under `label` that runs each of `circuits` for `shots` shots.
    fn _add_shot_loop(
        &mut self,
        label: &str,
        circuits: Vec<QuantumCircuitData<'_>>,
        shots: usize,
    ) -> PyResult<()> {
        let circuit_data: Vec<_> = circuits.into_iter().map(|c| c.data).collect();
        self.inner
            .add_node(label, ShotLoop::new(circuit_data, shots))
            .map_err(to_py_err)
    }

    /// Add a `ParameterExpressions` node under `label` that evaluates `expressions` from input
    /// values given for `parameters`, in that order.
    fn _add_parameter_expressions(
        &mut self,
        label: &str,
        expressions: Vec<PyParameterExpression>,
        parameters: Vec<Symbol>,
    ) -> PyResult<()> {
        let node = make_parameter_expressions(expressions, parameters)?;
        self.inner.add_node(label, node).map_err(to_py_err)
    }

    /// Add a directed edge from one node's output port to another's input port. Ports are
    /// `(label, path)` pairs; `path` is a list of `int`/`str` entries.
    fn _add_edge(
        &mut self,
        from_label: &str,
        from_path: Vec<Bound<'_, PyAny>>,
        to_label: &str,
        to_path: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let from = Port::new(from_label, parse_path(&from_path)?);
        let to = Port::new(to_label, parse_path(&to_path)?);
        self.inner.add_edge(from, to).map_err(to_py_err)
    }

    /// Declare a program output under `key`, bound to the `(from_label, from_path)` port.
    fn _set_output(
        &mut self,
        key: &str,
        from_label: &str,
        from_path: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let port = Port::new(from_label, parse_path(&from_path)?);
        self.inner.set_output(key, port).map_err(to_py_err)
    }

    /// The labels of every node in the graph, sorted (for deterministic introspection in
    /// tests of fan-out / dead-code elimination).
    fn _node_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .inner
            .iter_nodes()
            .map(|(l, _)| l.to_string())
            .collect();
        labels.sort();
        labels
    }

    /// The declared program input keys, in declaration order.
    fn input_keys(&self) -> Vec<String> {
        self.inner
            .input_types()
            .iter_children()
            .filter_map(|(k, _)| k.map(str::to_string))
            .collect()
    }

    /// The declared program output keys, in declaration order.
    fn output_keys(&self) -> Vec<String> {
        self.inner
            .output_types()
            .iter_children()
            .filter_map(|(k, _)| k.map(str::to_string))
            .collect()
    }

    /// Return the resolved `TensorSpec` of every declared output, keyed by output key.
    ///
    /// Runs the whole-graph `resolve_types_flat` pass (fed with the concrete types declared
    /// at input-time) rather than reading `output_types()` directly, since the latter holds
    /// each node's unresolved template type (e.g. `mean`'s output dtype is a symbolic
    /// variable until resolved against its actual input).
    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let input_types: Vec<TensorType> =
            self.inner.input_types().iter_leaves().cloned().collect();
        let resolved = self
            .inner
            .resolve_types_flat(&input_types)
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for (key, ty) in self.output_keys().into_iter().zip(resolved) {
            dict.set_item(key, tensor_type_to_spec(&ty)?)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumProgram(inputs={:?}, outputs={:?})",
            self.input_keys(),
            self.output_keys()
        )
    }
}

/// Register the `quantum_program` submodule.
pub fn quantum_program(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQuantumProgram>()?;
    m.add_class::<PyTensorSpec>()?;
    m.add_wrapped(wrap_pyfunction!(infer_op))?;
    m.add_wrapped(wrap_pyfunction!(infer_shot_loop))?;
    m.add_wrapped(wrap_pyfunction!(infer_parameter_expressions))?;
    m.add_wrapped(wrap_pyfunction!(spec_of_array))?;
    Ok(())
}
