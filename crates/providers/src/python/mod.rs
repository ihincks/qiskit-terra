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

//! Bindings for the `qiskit.quantum_program` Python package.

mod data_tree;

use pyo3::prelude::*;

pub use data_tree::PyDataTree;

/// Register the `quantum_program` submodule.
pub fn quantum_program(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataTree>()?;
    Ok(())
}
