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

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};
use qiskit_providers::{BackendV3, ExecutionResult, JobV2, QuantumProgram};
use qiskit_transpiler::target::Target;
use std::convert::Infallible;
use std::ffi::{CStr, c_char, c_void};

pub struct CJobInterface {
    context: *mut c_void,
    job_id: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
    status: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
    cancel: Option<unsafe extern "C" fn(context: *mut c_void)>,
    result: Option<unsafe extern "C" fn(context: *mut c_void) -> *const ExecutionResult>,
}

pub struct CBackendInterface {
    context: *mut c_void,
    //    init: Option<unsafe extern "C" fn(context: *mut c_void)>,
    name: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
    description: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
    target: Option<unsafe extern "C" fn(context: *mut c_void) -> *const Target>,
    execute_fn: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            program: *const QuantumProgram,
        ) -> *mut CJobInterface,
    >,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_backend_new(context: *mut c_void) -> *mut CBackendInterface {
    Box::into_raw(Box::new(CBackendInterface {
        context,
        name: None,
        description: None,
        target: None,
        execute_fn: None,
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_backend_set_target(
    backend: *mut CBackendInterface,
    fn_ptr: unsafe extern "C" fn(context: *mut c_void) -> *const Target,
) {
    let backend = unsafe { mut_ptr_as_ref(backend) };
    backend.target = Some(fn_ptr);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_backend_set_name(
    backend: *mut CBackendInterface,
    fn_ptr: unsafe extern "C" fn(context: *mut c_void) -> *const c_char,
) {
    let backend = unsafe { mut_ptr_as_ref(backend) };
    backend.name = Some(fn_ptr);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_backend_set_description(
    backend: *mut CBackendInterface,
    fn_ptr: unsafe extern "C" fn(context: *mut c_void) -> *const c_char,
) {
    let backend = unsafe { mut_ptr_as_ref(backend) };
    backend.description = Some(fn_ptr);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_backend_set_execute(
    backend: *mut CBackendInterface,
    fn_ptr: unsafe extern "C" fn(
        context: *mut c_void,
        program: *const QuantumProgram,
    ) -> *mut CJobInterface,
) {
    let backend = unsafe { mut_ptr_as_ref(backend) };
    backend.execute_fn = Some(fn_ptr);
}

impl BackendV3 for CBackendInterface {
    type Error = Infallible;

    fn name(&self) -> Option<&str> {
        self.name.and_then(|callback| {
            let result = unsafe { callback(self.context) };
            if result.is_null() {
                None
            } else {
                Some(unsafe { CStr::from_ptr(result) }.to_str().unwrap())
            }
        })
    }

    fn description(&self) -> Option<&str> {
        self.description.and_then(|callback| {
            let result = unsafe { callback(self.context) };
            if result.is_null() {
                None
            } else {
                Some(unsafe { CStr::from_ptr(result) }.to_str().unwrap())
            }
        })
    }

    fn target(&mut self) -> Result<&Target, Infallible> {
        let Some(callback) = self.target else {
            panic!(
                "target not set for backend interface, qk_backend_set_target must be set before using the backend"
            );
        };
        let target_ptr = unsafe { callback(self.context) };

        Ok(unsafe { const_ptr_as_ref(target_ptr) })
    }

    fn execute(&self, program: &QuantumProgram) -> impl JobV2<Error = Infallible> {
        let Some(callback) = self.execute_fn else {
            panic!(
                "execute function not set for backend interface. qk_backend_set_execute must be set before using the backend"
            );
        };
        let job_ptr = unsafe { callback(self.context, program) };
        unsafe { mut_ptr_as_ref(job_ptr) }
    }
}

impl JobV2 for &mut CJobInterface {
    type Error = Infallible;
    fn job_id(&self) -> &str {
        let Some(callback) = self.job_id else {
            panic!(
                "job id function not set for job interface. qk_job_set_job_id must be set before using this job"
            );
        };
        let result = unsafe { callback(self.context) };
        unsafe { CStr::from_ptr(result) }.to_str().unwrap()
    }

    fn status(&self) -> String {
        let Some(callback) = self.status else {
            panic!(
                "status function not set for job interface. qk_job_set_job_status must be set before using this job"
            );
        };
        let result = unsafe { callback(self.context) };
        unsafe { CStr::from_ptr(result) }
            .to_str()
            .unwrap()
            .to_owned()
    }

    fn cancel(&self) {
        let Some(callback) = self.cancel else {
            panic!(
                "cancel function not set for job interface. qk_job_set_job_cancel must be set before using this job"
            );
        };
        unsafe { callback(self.context) };
    }

    fn result(&self) -> Result<ExecutionResult, Infallible> {
        let Some(callback) = self.result else {
            panic!(
                "result function not set for job interface. qk_job_set_job_cancel must be set before using this job"
            );
        };
        let result = unsafe { const_ptr_as_ref(callback(self.context)) };
        Ok(result.clone())
    }
}
