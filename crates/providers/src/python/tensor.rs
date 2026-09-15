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

//! The Python binding of the tensor domain: dtypes, tensor types, and tensor values.
//!
//! A tensor crosses the boundary as a NumPy array. Reading one copies the buffer. Writing one
//! either hands the buffer over, which `tensor_object` does, or lends it out read-only, which
//! `tensor_view` does. A `Bit` tensor is a NumPy array of `bool`, and is copied in every direction,
//! because NumPy spells a bit `bool` and a tensor holds bytes.

use ndarray::ArrayD;
use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyTuple};

use crate::tensor::{DType, Dim, Tensor, TensorType};

#[pymethods]
impl DType {
    /// Return a [`TensorType`] of this dtype whose shape is `shape`.
    fn __getitem__(&self, shape: &Bound<'_, PyAny>) -> PyResult<TensorType> {
        let shape = match shape.cast::<PyTuple>() {
            Ok(axes) => axes
                .iter()
                .map(|axis| dim(&axis))
                .collect::<PyResult<_>>()?,
            Err(_) => vec![dim(shape)?],
        };
        Ok(TensorType {
            dtype: *self,
            shape,
        })
    }
}

/// An axis whose size is not known until run time, but is provably at most `max`.
#[pyclass(name = "bounded", module = "qiskit.quantum_program", frozen, eq, hash)]
#[derive(PartialEq, Eq, Hash)]
pub struct PyBounded {
    /// The largest size this axis can have.
    #[pyo3(get)]
    max: usize,
}

#[pymethods]
impl PyBounded {
    #[new]
    #[pyo3(signature = (max, /))]
    fn new(max: usize) -> Self {
        Self { max }
    }

    fn __repr__(&self) -> String {
        format!("bounded({})", self.max)
    }
}

#[pymethods]
impl TensorType {
    #[new]
    #[pyo3(signature = (dtype, shape, /))]
    fn new(dtype: DType, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            dtype,
            shape: parse_shape(shape)?,
        })
    }

    /// The element type.
    #[getter]
    fn dtype(&self) -> DType {
        self.dtype
    }

    /// The size of each axis, an integer or a `bounded()`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        shape_object(py, &self.shape)
    }

    fn __repr__(&self) -> String {
        format!("TensorType({self})")
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

/// Return the shape `object` describes, taking each of its items as one axis.
pub(super) fn parse_shape(object: &Bound<'_, PyAny>) -> PyResult<Vec<Dim>> {
    object.try_iter()?.map(|axis| dim(&axis?)).collect()
}

/// Return `shape` as Python spells it, one item per axis.
pub(super) fn shape_object<'py>(py: Python<'py>, shape: &[Dim]) -> PyResult<Bound<'py, PyTuple>> {
    let shape = shape
        .iter()
        .map(|&dim| dim_object(py, dim))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, shape)
}

/// Return the dimension `object` describes: an integer size, or a `bounded()` axis.
fn dim(object: &Bound<'_, PyAny>) -> PyResult<Dim> {
    if let Ok(bounded) = object.cast::<PyBounded>() {
        return Ok(Dim::Bounded {
            max: bounded.get().max,
        });
    }
    if let Ok(size) = object.extract::<usize>() {
        return Ok(Dim::Fixed(size));
    }
    if object.cast::<PyInt>().is_ok() {
        return Err(PyValueError::new_err(format!(
            "an axis cannot have size {}",
            object.str()?
        )));
    }
    Err(PyTypeError::new_err(format!(
        "an axis is sized by an integer or by bounded(), not by {}",
        object.get_type().name()?
    )))
}

/// Return `dim` as an integer for a fixed axis or a `bounded()` for a bounded one.
fn dim_object(py: Python<'_>, dim: Dim) -> PyResult<Py<PyAny>> {
    match dim {
        Dim::Fixed(size) => Ok(size.into_pyobject(py)?.into_any().unbind()),
        Dim::Bounded { max } => Ok(Py::new(py, PyBounded { max })?.into_any()),
    }
}

/// Read `object` as a tensor, converting it with `numpy.asarray` first.
///
/// The dtype of the resulting array is the dtype of the tensor: nothing is converted to a dtype a
/// caller has not asked for.
pub(super) fn tensor(object: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let py = object.py();
    let array = py
        .import("numpy")?
        .call_method1("asarray", (object,))?
        .into_any();

    // A `Bit` tensor is stored as `u8`, which numpy spells `bool`.
    if let Ok(bits) = array.cast::<PyArrayDyn<bool>>() {
        let bits = bits.readonly().as_array().mapv(u8::from);
        return Ok(Tensor::Bit(bits.into_shared()));
    }
    macro_rules! read {
        ($($variant:ident($element:ty)),* $(,)?) => {
            $(
                if let Ok(array) = array.cast::<PyArrayDyn<$element>>() {
                    let array: ArrayD<$element> = array.readonly().as_array().to_owned();
                    return Ok(Tensor::$variant(array.into_shared()));
                }
            )*
        };
    }
    read!(
        C64(Complex32),
        C128(Complex64),
        F32(f32),
        F64(f64),
        I8(i8),
        I16(i16),
        I32(i32),
        I64(i64),
        U8(u8),
        U16(u16),
        U32(u32),
        U64(u64),
    );
    Err(PyTypeError::new_err(format!(
        "a tensor cannot hold numpy dtype {}",
        array.getattr("dtype")?
    )))
}

/// Wrap `tensor` as a NumPy array over the tensor's own buffer, which `container` keeps alive.
///
/// The array is not writeable, so nothing done to it from Python reaches the tensor.
///
/// # Safety
///
/// `container` must own `tensor`'s buffer and keep it alive, in place and unwritten, for as long as
/// `container` itself lives. A frozen pyclass holding the tensor satisfies this: no mutable
/// reference to the buffer can be taken through it, and dropping the buffer means dropping the
/// object, which Python keeps from happening while the array holds a reference to it.
pub(super) unsafe fn tensor_view<'py>(
    tensor: &Tensor,
    container: Bound<'py, PyAny>,
) -> Bound<'py, PyAny> {
    macro_rules! view {
        ($($variant:ident($element:ty)),* $(,)?) => {
            match tensor {
                // NumPy spells a bit `bool`, which is a byte this one does not hold, so this arm
                // copies rather than lending anything out. The copy is made read-only too, which
                // leaves every array here read-only whatever the dtype.
                Tensor::Bit(bits) => {
                    let array = bits.mapv(|bit| bit != 0).into_pyarray(container.py());
                    array.readwrite().make_nonwriteable();
                    array.into_any()
                }
                $(Tensor::$variant(array) => {
                    // SAFETY: the caller guarantees that `container` owns this buffer and keeps it
                    // alive and unwritten, which is this function's contract.
                    let array = unsafe {
                        PyArrayDyn::<$element>::borrow_from_array(array, container)
                    };
                    array.readwrite().make_nonwriteable();
                    array.into_any()
                },)*
            }
        };
    }
    view!(
        C64(Complex32),
        C128(Complex64),
        F32(f32),
        F64(f64),
        I8(i8),
        I16(i16),
        I32(i32),
        I64(i64),
        U8(u8),
        U16(u16),
        U32(u32),
        U64(u64),
    )
}

/// Convert `tensor` to a NumPy array.
///
/// NumPy takes the buffer where the tensor is the only holder of it, and a copy of it otherwise.
pub(super) fn tensor_object(py: Python<'_>, tensor: Tensor) -> Py<PyAny> {
    macro_rules! write {
        ($($variant:ident),* $(,)?) => {
            match tensor {
                // NumPy spells a bit `bool`, which is a byte this one does not hold, so this arm
                // converts rather than handing anything over.
                Tensor::Bit(bits) => bits.mapv(|bit| bit != 0).to_pyarray(py).into_any().unbind(),
                $(Tensor::$variant(array) => match array.try_into_owned_nocopy() {
                    Ok(owned) => owned.into_pyarray(py).into_any().unbind(),
                    Err(shared) => shared.to_pyarray(py).into_any().unbind(),
                },)*
            }
        };
    }
    write!(C64, C128, F32, F64, I8, I16, I32, I64, U8, U16, U32, U64)
}
