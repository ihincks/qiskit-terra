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

use qiskit_transpiler::target::Target;

#[derive(Clone)]
pub struct QuantumProgram {
    pub placeholder: Vec<u8>,
}

#[derive(Clone)]
pub struct ExecutionResult {
    pub placeholder: Vec<u32>,
}

/// This is the trait defining the interface for job objects in the backend interface
pub trait JobV2 {
    type Error;

    fn job_id(&self) -> &str;
    fn result(&self) -> Result<ExecutionResult, Self::Error>;
    fn cancel(&self);
    fn status(&self) -> String;
}

/// This trait defines the common backend
pub trait BackendV3 {
    type Error;

    /// An optional name for this backend instance
    fn name(&self) -> Option<&str>;
    /// An optional description for this backend instance
    fn description(&self) -> Option<&str>;
    /// The target describing the QPU constraints for executing circuits
    fn target(&mut self) -> Result<&Target, Self::Error>;
    /// The execution
    fn execute(&self, program: &QuantumProgram) -> impl JobV2<Error = Self::Error>;
}
