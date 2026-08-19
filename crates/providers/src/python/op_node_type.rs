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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::chain;
use super::tensor::{parse_shape, tensor};
use crate::nodes::{
    Add, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BoxedOpNodeType, BroadcastTo, Cast,
    Constant, Divide, Mean, Multiply, Parity, Power, Remainder, Std, Subtract, Variance, erase,
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
pub struct PyOpNodeType(pub(super) BoxedOpNodeType);

#[pymethods]
impl PyOpNodeType {
    #[staticmethod]
    fn add() -> Self {
        Self(erase(Add))
    }

    #[staticmethod]
    fn subtract() -> Self {
        Self(erase(Subtract))
    }

    #[staticmethod]
    fn multiply() -> Self {
        Self(erase(Multiply))
    }

    #[staticmethod]
    fn divide() -> Self {
        Self(erase(Divide))
    }

    #[staticmethod]
    fn remainder() -> Self {
        Self(erase(Remainder))
    }

    #[staticmethod]
    fn power() -> Self {
        Self(erase(Power))
    }

    #[staticmethod]
    fn bitwise_and() -> Self {
        Self(erase(BitwiseAnd))
    }

    #[staticmethod]
    fn bitwise_or() -> Self {
        Self(erase(BitwiseOr))
    }

    #[staticmethod]
    fn bitwise_xor() -> Self {
        Self(erase(BitwiseXor))
    }

    #[staticmethod]
    fn bitwise_not() -> Self {
        Self(erase(BitwiseNot))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, /))]
    fn mean(axis: usize) -> Self {
        Self(erase(Mean::new(axis)))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, ddof, /))]
    fn variance(axis: usize, ddof: f64) -> Self {
        Self(erase(Variance::new(axis, ddof)))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, ddof, /))]
    fn std(axis: usize, ddof: f64) -> Self {
        Self(erase(Std::new(axis, ddof)))
    }

    #[staticmethod]
    #[pyo3(signature = (axis, /))]
    fn parity(axis: usize) -> Self {
        Self(erase(Parity::new(axis)))
    }

    #[staticmethod]
    #[pyo3(signature = (target, /))]
    fn cast(target: DType) -> Self {
        Self(erase(Cast::new(target)))
    }

    #[staticmethod]
    #[pyo3(signature = (target, /))]
    fn broadcast_to(target: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(erase(BroadcastTo::new(parse_shape(target)?))))
    }

    #[staticmethod]
    #[pyo3(signature = (value, /))]
    fn constant(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(erase(Constant::new(tensor(value)?))))
    }

    /// The type name, qualified by its namespace.
    #[getter]
    fn full_name(&self) -> String {
        self.0.full_name()
    }

    /// The types this operation produces from operands of type `operands`.
    fn output_types(&self, operands: Vec<TensorType>) -> PyResult<Vec<TensorType>> {
        self.0.infer_output_types(&operands).map_err(|error| {
            PyValueError::new_err(format!("{}: {}", self.0.full_name(), chain(&*error)))
        })
    }
}
