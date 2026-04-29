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

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;

use qiskit_providers::{BackendV3, ExecutionResult, JobV2, QuantumProgram};
use qiskit_transpiler::target::Target;
use qiskit_util::py::ImportOnceCell;

static BACKEND_V3_ABC: ImportOnceCell =
    ImportOnceCell::new("qiskit.providers.backend", "BackendV3");

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

#[pyclass(subclass, skip_from_py_object)]
pub struct BaseBackendV3 {
    #[pyo3(get, set)]
    name: Option<String>,
    #[pyo3(get, set)]
    description: Option<String>,
    #[pyo3(get, set)]
    target: Py<PyAny>,
    #[pyo3(get, set)]
    execute_fn: Py<PyAny>,
    _target: Option<Target>,
}

// TODO: Remove the clone dervive
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

impl<'py> FromPyObject<'_, 'py> for BaseBackendV3 {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        if obj.is_instance(BACKEND_V3_ABC.get_bound(py))? {
            Ok(BaseBackendV3 {
                name: obj.getattr(intern!(py, "name"))?.extract()?,
                description: obj.getattr(intern!(py, "description"))?.extract()?,
                target: obj.getattr(intern!(py, "target"))?.unbind(),
                _target: None,
                execute_fn: obj.getattr(intern!(py, "execute"))?.unbind(),
            })
        } else {
            Err(PyTypeError::new_err("Invalid type not a backend v3 object"))
        }
    }
}

#[pymethods]
impl BaseBackendV3 {
    #[new]
    pub fn new(
        name: Option<String>,
        description: Option<String>,
        target_fn: Py<PyAny>,
        execute_fn: Py<PyAny>,
    ) -> Self {
        BaseBackendV3 {
            name,
            description,
            target: target_fn,
            _target: None,
            execute_fn,
        }
    }
}

impl BackendV3 for BaseBackendV3 {
    type Error = PyErr;

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn target(&mut self) -> Result<&Target, Self::Error> {
        Python::attach(|py| -> PyResult<()> {
            let bound_fn = self.target.bind(py);
            self._target = Some(bound_fn.call0()?.extract()?);
            Ok(())
        })?;
        Ok(self._target.as_ref().unwrap())
    }

    fn execute(&self, program: &QuantumProgram) -> impl JobV2<Error = PyErr> {
        let program: QuantumProgram = program.clone();
        let out_program = PyQuantumProgram(program);
        Python::attach(|py| {
            let result = self.execute_fn.bind(py).call1((out_program,)).unwrap();
            result.extract::<BaseJobV2>().unwrap()
        })
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
