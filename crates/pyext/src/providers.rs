// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;

use qiskit_providers::{BackendV3, ExecutionResult, JobV2, QuantumProgram};
use qiskit_transpiler::target::Target;

#[derive(Clone)]
#[pyclass(name = "QuantumProgram", from_py_object)]
pub struct PyQuantumProgram(QuantumProgram);

#[pymethods]
impl PyQuantumProgram {
    pub fn len(&self) -> usize {
        // Demonstrate exposing the program methods to python
        self.0.placeholder.len()
    }
}

#[derive(Clone)]
#[pyclass(name = "ExecutionResult", from_py_object)]
pub struct PyExecutionResult(ExecutionResult);

/// Marker class. Python's `BackendV3` ABC inherits from this, which allows
/// Rust code to use `is_instance_of::<BaseBackendV3>()` as a typed
/// isinstance check without a string-based import.
#[pyclass(subclass, module = "qiskit._accelerate.providers")]
pub struct BaseBackendV3 {}

#[pymethods]
impl BaseBackendV3 {
    #[new]
    pub fn new() -> Self {
        BaseBackendV3 {}
    }
}

// TODO: Remove the clone derive
#[derive(Clone)]
#[pyclass(subclass, from_py_object)]
pub struct BaseJobV2 {
    #[pyo3(get, set)]
    job_id: String,
    #[pyo3(get, set)]
    cancel: Py<PyAny>,
    #[pyo3(get, set)]
    status: Py<PyAny>,
    #[pyo3(get, set)]
    result: Py<PyAny>,
}

/// Rust-internal adapter wrapping a Python `BackendV3` object. Implements the
/// `BackendV3` trait by calling into Python. `name` and `description` are
/// extracted eagerly; `target` is fetched once and cached via `OnceLock`.
pub struct PyBackend {
    obj: Py<PyAny>,
    name: Option<String>,
    description: Option<String>,
    _target: Option<Target>,
}

impl<'py> FromPyObject<'_, 'py> for PyBackend {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        if obj.is_instance_of::<BaseBackendV3>() {
            Ok(PyBackend {
                name: obj.getattr(intern!(py, "name"))?.extract()?,
                description: obj.getattr(intern!(py, "description"))?.extract()?,
                _target: None,
                obj: obj.to_owned().unbind(),
            })
        } else {
            Err(PyTypeError::new_err(
                "Expected a BackendV3 instance (must inherit from BaseBackendV3)",
            ))
        }
    }
}

impl BackendV3 for PyBackend {
    type Error = PyErr;

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn target(&mut self) -> Result<&Target, Self::Error> {
        if self._target.is_none() {
            self._target = Some(Python::attach(|py| -> PyResult<Target> {
                self.obj
                    .bind(py)
                    .getattr(intern!(py, "target"))?
                    .extract()
                    .map_err(PyErr::from)
            })?);
        }
        Ok(self._target.as_ref().unwrap())
    }

    fn execute(&self, program: &QuantumProgram) -> impl JobV2<Error = PyErr> {
        let py_program = PyQuantumProgram(program.clone());
        Python::attach(|py| {
            let result = self
                .obj
                .bind(py)
                .call_method1(intern!(py, "execute"), (py_program,))?;
            result.extract::<BaseJobV2>().map_err(PyErr::from)
        })
        .unwrap()
    }
}

impl JobV2 for BaseJobV2 {
    type Error = PyErr;
    fn job_id(&self) -> &str {
        self.job_id.as_str()
    }

    fn status(&self) -> String {
        Python::attach(|py: Python| {
            let bound_fn = self.status.bind(py);
            let out = bound_fn.call0().unwrap();
            out.extract().unwrap()
        })
    }

    fn result(&self) -> Result<ExecutionResult, Self::Error> {
        Python::attach(|py| {
            let result: PyExecutionResult = self.result.bind(py).call0()?.extract()?;
            Ok(result.0)
        })
    }

    fn cancel(&self) {
        Python::attach(|py| {
            let _ = self.cancel.bind(py).call0();
        })
    }
}

pub fn providers(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyQuantumProgram>()?;
    m.add_class::<PyExecutionResult>()?;
    m.add_class::<BaseBackendV3>()?;
    m.add_class::<BaseJobV2>()?;
    Ok(())
}
