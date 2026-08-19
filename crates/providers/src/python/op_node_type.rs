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

//! The Python binding of the op node catalogue.

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::parameter::parameter_expression::{PyParameter, PyParameterExpression};
use qiskit_circuit::parameter::symbol_expr::Symbol;

use super::data_tree::PyDataTree;
use super::tensor::{parse_shape, tensor};
use super::{chain, value_error};
use crate::data_tree::DataTree;
use crate::nodes::{
    Add, BindParameters, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BoxedOpNodeType,
    BroadcastTo, Cast, Constant, Divide, Mean, Multiply, Parity, Power, Remainder, ShotLoop, Std,
    Subtract, Variance, erase,
};
use crate::tensor::{DType, TensorType};

/// One of the operations a node can perform, applied by adding a node to a program function.
///
/// A node type has whatever the operation needs beyond its operands, such as the axis a
/// reduction folds along, and it reports the types it produces from the types it is given.
#[pyclass(
    name = "OpNodeType",
    module = "qiskit._accelerate.quantum_program",
    frozen
)]
pub struct PyOpNodeType {
    pub(super) node_type: BoxedOpNodeType,
    /// How this operation arranges its results, one leaf per result in the order it produces them.
    structure: DataTree<()>,
}

impl PyOpNodeType {
    /// A node type producing one result.
    fn single(node_type: BoxedOpNodeType) -> Self {
        Self {
            node_type,
            structure: DataTree::new_leaf(()),
        }
    }
}

#[pymethods]
impl PyOpNodeType {
    #[staticmethod]
    fn add() -> Self {
        Self::single(erase(Add))
    }

    #[staticmethod]
    fn subtract() -> Self {
        Self::single(erase(Subtract))
    }

    #[staticmethod]
    fn multiply() -> Self {
        Self::single(erase(Multiply))
    }

    #[staticmethod]
    fn divide() -> Self {
        Self::single(erase(Divide))
    }

    #[staticmethod]
    fn remainder() -> Self {
        Self::single(erase(Remainder))
    }

    #[staticmethod]
    fn power() -> Self {
        Self::single(erase(Power))
    }

    #[staticmethod]
    fn bitwise_and() -> Self {
        Self::single(erase(BitwiseAnd))
    }

    #[staticmethod]
    fn bitwise_or() -> Self {
        Self::single(erase(BitwiseOr))
    }

    #[staticmethod]
    fn bitwise_xor() -> Self {
        Self::single(erase(BitwiseXor))
    }

    #[staticmethod]
    fn bitwise_not() -> Self {
        Self::single(erase(BitwiseNot))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, /))]
    fn mean(axis: usize) -> Self {
        Self::single(erase(Mean::new(axis)))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, ddof, /))]
    fn variance(axis: usize, ddof: f64) -> Self {
        Self::single(erase(Variance::new(axis, ddof)))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, ddof, /))]
    fn std(axis: usize, ddof: f64) -> Self {
        Self::single(erase(Std::new(axis, ddof)))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, /))]
    fn parity(axis: usize) -> Self {
        Self::single(erase(Parity::new(axis)))
    }

    #[staticmethod]
    #[pyo3(signature = (target, /))]
    fn cast(target: DType) -> Self {
        Self::single(erase(Cast::new(target)))
    }

    #[staticmethod]
    #[pyo3(signature = (target, /))]
    fn broadcast_to(target: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::single(erase(BroadcastTo::new(parse_shape(target)?))))
    }

    #[staticmethod]
    #[pyo3(signature = (value, /))]
    fn constant(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::single(erase(Constant::new(tensor(value)?))))
    }

    #[staticmethod]
    #[pyo3(signature = (circuits, shots, /))]
    fn shot_loop(circuits: Vec<Bound<'_, PyAny>>, shots: usize) -> PyResult<Self> {
        let circuits = circuits
            .iter()
            .enumerate()
            .map(|(index, circuit)| circuit_data(index, circuit))
            .collect::<PyResult<Vec<_>>>()?;
        let node = ShotLoop::new(circuits, shots).map_err(|error| value_error(&error))?;
        let structure = node.output_structure().clone();
        Ok(Self {
            node_type: erase(node),
            structure,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (expressions, parameters, /))]
    fn bind_parameters(
        expressions: Vec<Bound<'_, PyAny>>,
        parameters: Vec<PyParameter>,
    ) -> PyResult<Self> {
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
        let node =
            BindParameters::new(expressions, parameters).map_err(|error| value_error(&error))?;
        Ok(Self::single(erase(node)))
    }

    /// The type name, qualified by its namespace.
    #[getter]
    fn full_name(&self) -> String {
        self.node_type.full_name()
    }

    /// The types this operation produces from operands of type `operands`, arranged as it produces
    /// them.
    fn output_types(&self, py: Python<'_>, operands: Vec<TensorType>) -> PyResult<PyDataTree> {
        let arity = self.node_type.arity();
        if operands.len() != arity {
            return Err(PyValueError::new_err(format!(
                "{} takes {arity} operands, got {}",
                self.node_type.full_name(),
                operands.len()
            )));
        }
        let types = self
            .node_type
            .infer_output_types(&operands)
            .map_err(|error| {
                PyValueError::new_err(format!(
                    "{}: {}",
                    self.node_type.full_name(),
                    chain(&*error)
                ))
            })?;
        let types = types
            .into_iter()
            .map(|ty| Ok(Py::new(py, ty)?.into_any()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyDataTree(self.structure.unflatten(types).expect(
            "an operation produces one result per leaf of its output structure",
        )))
    }
}

/// The data of the `index`th circuit given, copied out of it.
///
/// A circuit that assembles itself on demand does so when its `data` is read, which is why that is
/// read before the data itself is taken.
fn circuit_data(index: usize, circuit: &Bound<'_, PyAny>) -> PyResult<Arc<CircuitData>> {
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
    Ok(Arc::new(CircuitData::clone(&data.borrow())))
}
