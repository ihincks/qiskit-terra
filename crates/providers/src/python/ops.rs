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

//! The Python binding of the op catalogue, one class per operation.
//!
//! Every class here extends `ProgramOp`, which is also what an op defined outside Qiskit reads back
//! as. A class builds an operation and reads the payload it was built with, so one class serves both
//! the author of a program and the reader of one.

use pyo3::PyClass;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::imports::QUANTUM_CIRCUIT;
use qiskit_circuit::parameter::parameter_expression::{
    ParameterExpression, PyParameter, PyParameterExpression,
};
use qiskit_circuit::parameter::symbol_expr::Symbol;

use super::data_tree::{ObjectTree, PyDataTree};
use super::tensor::{parse_shape, shape_object, tensor, tensor_view};
use super::{chain, position, value_error};
use crate::InvalidName;
use crate::data_tree::DataTree;
use crate::ops::{
    Add, BindParameters, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BoxedProgramOp,
    BroadcastTo, Cast, Constant, Divide, Mean, Multiply, Parity, Power, ProgramOp, Remainder,
    ShotLoop, Std, Subtract, Variance,
};
use crate::tensor::{DType, TensorType};

/// One of the operations an instruction can perform, applied by adding an instruction to a program
/// function.
///
/// An op has whatever the operation needs beyond its operands, such as the axis a reduction folds
/// along, and it reports the types it produces from the types it is given. Each operation Qiskit
/// defines has a class of its own, and this is the base of all of them. An op defined outside Qiskit
/// reads back as this base, which reports its name and a summary of its payload and nothing else.
#[pyclass(
    name = "ProgramOp",
    module = "qiskit.quantum_program.ops",
    subclass,
    frozen
)]
pub struct PyProgramOp {
    pub(super) op: BoxedProgramOp,
}

impl PyProgramOp {
    /// Return the op this holds, which is always of type `O`.
    ///
    /// Each class of the catalogue builds its base with an op of one type, and every class here is
    /// frozen, so the type it was built with is the type it holds.
    fn payload<O: ProgramOp + 'static>(&self) -> &O {
        self.op
            .downcast_ref()
            .expect("a class of the catalogue holds an op of its own type")
    }

    /// Arrange `values`, one per result this operation produces, as it arranges its results.
    ///
    /// A shot loop is the one operation that arranges its results, and it says how. One result is a
    /// leaf, and several are a sequence.
    fn arrange(&self, values: Vec<Py<PyAny>>) -> Result<ObjectTree, InvalidName> {
        if let Some(shot_loop) = self.op.downcast_ref::<ShotLoop>() {
            let mut values = values.into_iter();
            let circuit_outputs = shot_loop
                .circuits()
                .iter()
                .map(|circuit| {
                    DataTree::mapping(circuit.cregs().iter().map(|creg| {
                        let value = values
                            .next()
                            .expect("one value per register of each circuit");
                        (creg.name(), DataTree::new_leaf(value))
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(DataTree::sequence(circuit_outputs));
        }
        match <[Py<PyAny>; 1]>::try_from(values) {
            Ok([value]) => Ok(DataTree::new_leaf(value)),
            Err(values) => Ok(DataTree::sequence(
                values.into_iter().map(DataTree::new_leaf),
            )),
        }
    }
}

#[pymethods]
impl PyProgramOp {
    /// The type name, qualified by its namespace.
    #[getter]
    fn full_name(&self) -> String {
        self.op.full_name()
    }

    /// A summary of the payload, such as ``axis=0``, and ``None`` for an operation with none.
    #[getter]
    fn describe(&self) -> Option<String> {
        self.op.describe()
    }

    /// Return the types this operation produces from operands of type ``operands``, arranged as it
    /// produces them.
    ///
    /// Args:
    ///     operands: The type of each operand, one per operand the operation takes.
    ///
    /// Returns:
    ///     A data tree of the types produced, which is one leaf for an operation producing one
    ///     value.
    ///
    /// Raises:
    ///     ValueError: If there is not one type per operand, or if the operation does not accept the
    ///         types given.
    #[pyo3(signature = (operands, /))]
    fn output_types(&self, py: Python<'_>, operands: Vec<TensorType>) -> PyResult<PyDataTree> {
        let arity = self.op.arity();
        if operands.len() != arity {
            return Err(PyValueError::new_err(format!(
                "{} takes {arity} operands, got {}",
                self.op.full_name(),
                operands.len()
            )));
        }
        let types = self.op.infer_output_types(&operands).map_err(|error| {
            PyValueError::new_err(format!("{}: {}", self.op.full_name(), chain(&*error)))
        })?;
        let types = types
            .into_iter()
            .map(|ty| Ok(Py::new(py, ty)?.into_any()))
            .collect::<PyResult<Vec<_>>>()?;
        let types = self.arrange(types).map_err(|error| value_error(&error))?;
        Ok(PyDataTree(types))
    }
}

/// Initialize `class`, one of the classes of the catalogue, over the op `op`.
///
/// The op goes on the base class, so a class of the catalogue holds nothing of its own and is a name
/// plus the getters that read that op.
fn init<C, O>(op: O, class: C) -> PyClassInitializer<C>
where
    C: PyClass<BaseType = PyProgramOp>,
    O: ProgramOp + Clone + Send + Sync + 'static,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    init_boxed(Box::new(op), class)
}

/// Initialize `class`, one of the classes of the catalogue, over the op `op` already boxed.
fn init_boxed<C>(op: BoxedProgramOp, class: C) -> PyClassInitializer<C>
where
    C: PyClass<BaseType = PyProgramOp>,
{
    PyClassInitializer::from(PyProgramOp { op }).add_subclass(class)
}

/// Wrap `op` as the class that reads it.
///
/// An op defined outside this crate has no class of its own, so it reads back as the base class,
/// which reports its name and a summary of its payload and nothing else.
pub(super) fn op_object<'py>(py: Python<'py>, op: BoxedProgramOp) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! known {
        ($($op:ty => $class:expr),* $(,)?) => {
            $(if op.downcast_ref::<$op>().is_some() {
                return Ok(Bound::new(py, init_boxed(op, $class))?.into_any());
            })*
        };
    }
    known!(
        Add => PyAdd,
        Subtract => PySubtract,
        Multiply => PyMultiply,
        Divide => PyDivide,
        Remainder => PyRemainder,
        Power => PyPower,
        BitwiseAnd => PyBitwiseAnd,
        BitwiseOr => PyBitwiseOr,
        BitwiseXor => PyBitwiseXor,
        BitwiseNot => PyBitwiseNot,
        Parity => PyParity,
        Mean => PyMean,
        Variance => PyVariance,
        Std => PyStd,
        Cast => PyCast,
        BroadcastTo => PyBroadcastTo,
        Constant => PyConstant,
        ShotLoop => PyShotLoop,
        BindParameters => PyBindParameters,
    );
    Ok(Bound::new(py, PyProgramOp { op })?.into_any())
}

/// Define the class of an operation that has no payload of its own.
macro_rules! payload_free_op {
    ($(#[$documentation:meta])* $class:ident, $name:literal, $op:expr) => {
        $(#[$documentation])*
        #[pyclass(
            name = $name,
            module = "qiskit.quantum_program.ops",
            extends = PyProgramOp,
            frozen
        )]
        pub struct $class;

        #[pymethods]
        impl $class {
            #[new]
            fn new() -> PyClassInitializer<Self> {
                init($op, Self)
            }
        }
    };
}

payload_free_op!(
    /// Add two tensors elementwise.
    ///
    /// The operands promote to a common dtype and broadcast to a common shape, as they do for every
    /// elementwise operation.
    PyAdd,
    "Add",
    Add
);

payload_free_op!(
    /// Subtract the second tensor from the first, elementwise.
    PySubtract,
    "Subtract",
    Subtract
);

payload_free_op!(
    /// Multiply two tensors elementwise.
    PyMultiply,
    "Multiply",
    Multiply
);

payload_free_op!(
    /// Divide the first tensor by the second, elementwise.
    ///
    /// A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.
    PyDivide,
    "Divide",
    Divide
);

payload_free_op!(
    /// Remainder of the first tensor divided by the second, elementwise.
    ///
    /// A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.
    PyRemainder,
    "Remainder",
    Remainder
);

payload_free_op!(
    /// Raise the first tensor to the power of the second, elementwise.
    PyPower,
    "Power",
    Power
);

payload_free_op!(
    /// Bitwise AND of two bit-valued tensors, elementwise.
    PyBitwiseAnd,
    "BitwiseAnd",
    BitwiseAnd
);

payload_free_op!(
    /// Bitwise OR of two bit-valued tensors, elementwise.
    PyBitwiseOr,
    "BitwiseOr",
    BitwiseOr
);

payload_free_op!(
    /// Bitwise XOR of two bit-valued tensors, elementwise.
    PyBitwiseXor,
    "BitwiseXor",
    BitwiseXor
);

payload_free_op!(
    /// Bitwise NOT of a bit-valued tensor, elementwise.
    PyBitwiseNot,
    "BitwiseNot",
    BitwiseNot
);

/// XOR-reduce the bits of a tensor along one axis, removing that axis.
///
/// The parity of a sequence of bits is 1 when an odd number of them are 1, and 0 otherwise.
///
/// Args:
///     axis: The axis to XOR-reduce along.
#[pyclass(
    name = "Parity",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyParity;

#[pymethods]
impl PyParity {
    #[new]
    #[pyo3(signature = (axis, /))]
    fn new(axis: usize) -> PyClassInitializer<Self> {
        init(Parity::new(axis), Self)
    }

    /// The axis this XOR-reduces along.
    #[getter]
    fn axis(slf: &Bound<'_, Self>) -> usize {
        slf.as_super().get().payload::<Parity>().axis()
    }
}

/// Average a tensor along one axis, removing that axis.
///
/// Averaging bits or integers gives a float.
///
/// Args:
///     axis: The axis to average along.
#[pyclass(
    name = "Mean",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyMean;

#[pymethods]
impl PyMean {
    #[new]
    #[pyo3(signature = (axis, /))]
    fn new(axis: usize) -> PyClassInitializer<Self> {
        init(Mean::new(axis), Self)
    }

    /// The axis this averages along.
    #[getter]
    fn axis(slf: &Bound<'_, Self>) -> usize {
        slf.as_super().get().payload::<Mean>().axis()
    }
}

/// Variance of a tensor along one axis, removing that axis.
///
/// The sum of squared deviations is divided by ``n - ddof``, where ``n`` is the length of the axis.
///
/// Args:
///     axis: The axis to take the variance along.
///     ddof: The delta degrees of freedom, subtracted from the divisor.
#[pyclass(
    name = "Variance",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyVariance;

#[pymethods]
impl PyVariance {
    #[new]
    #[pyo3(signature = (axis, /, ddof = 0.0))]
    fn new(axis: usize, ddof: f64) -> PyClassInitializer<Self> {
        init(Variance::new(axis, ddof), Self)
    }

    /// The axis this takes the variance along.
    #[getter]
    fn axis(slf: &Bound<'_, Self>) -> usize {
        slf.as_super().get().payload::<Variance>().axis()
    }

    /// The delta degrees of freedom, subtracted from the divisor.
    #[getter]
    fn ddof(slf: &Bound<'_, Self>) -> f64 {
        slf.as_super().get().payload::<Variance>().ddof()
    }
}

/// Standard deviation of a tensor along one axis, removing that axis.
///
/// This is the square root of the variance, and ``ddof`` means the same thing.
///
/// Args:
///     axis: The axis to take the standard deviation along.
///     ddof: The delta degrees of freedom, subtracted from the divisor.
#[pyclass(
    name = "Std",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyStd;

#[pymethods]
impl PyStd {
    #[new]
    #[pyo3(signature = (axis, /, ddof = 0.0))]
    fn new(axis: usize, ddof: f64) -> PyClassInitializer<Self> {
        init(Std::new(axis, ddof), Self)
    }

    /// The axis this takes the standard deviation along.
    #[getter]
    fn axis(slf: &Bound<'_, Self>) -> usize {
        slf.as_super().get().payload::<Std>().axis()
    }

    /// The delta degrees of freedom, subtracted from the divisor.
    #[getter]
    fn ddof(slf: &Bound<'_, Self>) -> f64 {
        slf.as_super().get().payload::<Std>().ddof()
    }
}

/// Reinterpret a tensor as another dtype, keeping its shape.
///
/// Args:
///     target: The dtype to cast to. A complex value cannot be cast to a real dtype.
#[pyclass(
    name = "Cast",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyCast;

#[pymethods]
impl PyCast {
    #[new]
    #[pyo3(signature = (target, /))]
    fn new(target: DType) -> PyClassInitializer<Self> {
        init(Cast::new(target), Self)
    }

    /// The dtype this casts to.
    #[getter]
    fn target(slf: &Bound<'_, Self>) -> DType {
        slf.as_super().get().payload::<Cast>().target()
    }
}

/// Broadcast a tensor to a shape, aligning its axes with the trailing axes of that shape.
///
/// Args:
///     target: The shape to reach. Each axis of the operand must either match the axis it aligns
///         with or have size one.
#[pyclass(
    name = "BroadcastTo",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyBroadcastTo;

#[pymethods]
impl PyBroadcastTo {
    #[new]
    #[pyo3(signature = (target, /))]
    fn new(target: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        Ok(init(BroadcastTo::new(parse_shape(target)?), Self))
    }

    /// The shape this broadcasts to.
    #[getter]
    fn target<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let base = slf.as_super().get();
        shape_object(slf.py(), base.payload::<BroadcastTo>().target())
    }
}

/// Supply a tensor the program holds, rather than one given at call time.
///
/// Args:
///     value: The value to hold, read with :func:`numpy.asarray`.
#[pyclass(
    name = "Constant",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyConstant;

#[pymethods]
impl PyConstant {
    #[new]
    #[pyo3(signature = (value, /))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        Ok(init(Constant::new(tensor(value)?), Self))
    }

    /// The tensor this supplies, as a read-only array over the op's own buffer.
    ///
    /// A bit-valued tensor is a copy, because NumPy spells a bit ``bool``.
    #[getter]
    fn value<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyAny> {
        let base = slf.as_super();
        // SAFETY: the base class owns the boxed op that owns this tensor, and both classes are
        // frozen, so no mutable reference to the buffer exists. The array holds a reference to the
        // object, so the buffer stays alive.
        unsafe {
            tensor_view(
                base.get().payload::<Constant>().value(),
                base.clone().into_any(),
            )
        }
    }
}

/// Run each of several circuits for a number of shots, over the values given for its parameters.
///
/// The operation takes one operand per circuit, holding that circuit's parameter values in its
/// trailing axis, and produces one result per classical register of each circuit, ordered by circuit
/// and then by register.
///
/// Args:
///     circuits: The circuits to run, each copied as it is wired in.
///     shots: How many shots to run each circuit for.
///
/// Raises:
///     TypeError: If an entry of ``circuits`` is not a circuit.
///     ValueError: If a register's name cannot name a value.
#[pyclass(
    name = "ShotLoop",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyShotLoop;

#[pymethods]
impl PyShotLoop {
    #[new]
    #[pyo3(signature = (circuits, shots, /))]
    fn new(circuits: Vec<Bound<'_, PyAny>>, shots: usize) -> PyResult<PyClassInitializer<Self>> {
        let circuits = circuits
            .iter()
            .enumerate()
            .map(|(index, circuit)| {
                let py = circuit.py();
                let data = circuit
                    .getattr(intern!(py, "data"))
                    .and_then(|_| circuit.getattr(intern!(py, "_data")))
                    .ok()
                    .and_then(|data| data.cast_into::<PyCircuitData>().ok());
                let Some(data) = data else {
                    return Err(PyTypeError::new_err(format!(
                        "circuit {index}: expected a QuantumCircuit, got {}",
                        circuit.get_type().name()?
                    )));
                };
                Ok(CircuitData::clone(&data.borrow()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let op = ShotLoop::new(circuits, shots).map_err(|error| value_error(&error))?;
        Ok(init(op, Self))
    }

    /// How many shots each circuit runs for.
    #[getter]
    fn shots(slf: &Bound<'_, Self>) -> usize {
        slf.as_super().get().payload::<ShotLoop>().shots()
    }

    /// How many circuits this runs, which is how many operands it takes.
    #[getter]
    fn num_circuits(slf: &Bound<'_, Self>) -> usize {
        slf.as_super().get().payload::<ShotLoop>().circuits().len()
    }

    /// Return the circuit at ``index``.
    ///
    /// Only a circuit's data is stored, so the circuit given back has neither the name nor the
    /// metadata of the one it was built from.
    ///
    /// Args:
    ///     index: Which circuit to read, counted from the end when negative.
    ///
    /// Returns:
    ///     A copy of that circuit.
    ///
    /// Raises:
    ///     IndexError: If ``index`` addresses no circuit.
    #[pyo3(signature = (index, /))]
    fn circuit<'py>(slf: &Bound<'py, Self>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let base = slf.as_super().get();
        let circuits = base.payload::<ShotLoop>().circuits();
        let index = position(index, circuits.len(), "circuit")?;
        quantum_circuit(slf.py(), &circuits[index])
    }

    /// Return every circuit this runs, in the order it takes its operands.
    ///
    /// Each call copies every circuit, as :meth:`circuit` copies one.
    ///
    /// Returns:
    ///     A copy of each circuit.
    fn circuits<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let base = slf.as_super().get();
        base.payload::<ShotLoop>()
            .circuits()
            .iter()
            .map(|data| quantum_circuit(slf.py(), data))
            .collect()
    }
}

/// Evaluate each of several parameter expressions over a batch of values for some parameters.
///
/// The operation takes one operand, holding one value per declared parameter in its trailing axis,
/// and produces one result holding each expression's value in its trailing axis. Expressions
/// evaluate in double precision, so the result is ``f64``.
///
/// Args:
///     expressions: The expressions to evaluate.
///     parameters: The parameters the values are for. Every parameter an expression references must
///         appear here, and surplus ones are ignored.
///
/// Raises:
///     ValueError: If an expression references a parameter ``parameters`` does not name.
#[pyclass(
    name = "BindParameters",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyBindParameters;

#[pymethods]
impl PyBindParameters {
    #[new]
    #[pyo3(signature = (expressions, parameters, /))]
    fn new(
        expressions: Vec<Bound<'_, PyAny>>,
        parameters: Vec<PyParameter>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let expressions = expressions
            .iter()
            .map(|expression| {
                PyParameterExpression::extract_coerce(expression.as_borrowed())
                    .map(|expression| expression.inner)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let parameters = parameters
            .iter()
            .map(|parameter| Symbol::clone(&parameter.0))
            .collect();
        let op =
            BindParameters::new(expressions, parameters).map_err(|error| value_error(&error))?;
        Ok(init(op, Self))
    }

    /// The expressions this evaluates, in the order it produces their values.
    #[getter]
    fn expressions<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let base = slf.as_super().get();
        base.payload::<BindParameters>()
            .expressions()
            .iter()
            .map(|expression| {
                Ok(ParameterExpression::clone(expression)
                    .into_pyobject(slf.py())?
                    .into_any())
            })
            .collect()
    }

    /// The parameters the values are for, in the order the operand holds them.
    #[getter]
    fn parameters<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let base = slf.as_super().get();
        base.payload::<BindParameters>()
            .parameters()
            .iter()
            .map(|symbol| PyParameter::from(Symbol::clone(symbol)).into_pyobject(slf.py()))
            .collect()
    }
}

/// Return a new Python QuantumCircuit whose data is a clone of `data`.
fn quantum_circuit<'py>(py: Python<'py>, data: &CircuitData) -> PyResult<Bound<'py, PyAny>> {
    let data = PyCircuitData::from(CircuitData::clone(data));
    QUANTUM_CIRCUIT
        .get_bound(py)
        .call_method1(intern!(py, "_from_circuit_data"), (data,))
}
