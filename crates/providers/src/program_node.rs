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

use crate::data_tree::{ArityMismatch, DataTree, TreeMatchError};
use crate::quantum_program::QuantumProgram;
use crate::tensor::{DType, Tensor, TensorType};
use thiserror::Error;

/// Destructure `$args: &[Tensor]` into the named bindings, returning
/// [`CallInputError::WrongArity`] if the slice length does not match the pattern.
///
/// ```ignore
/// crate::unpack_tensor_args!(args, [x, y]);   // expects exactly 2
/// crate::unpack_tensor_args!(args, [x]);      // expects exactly 1
/// ```
#[macro_export]
macro_rules! unpack_tensor_args {
    ($args:ident, [$($x:ident),+]) => {
        let [$($x),+] = $args else {
            return Err($crate::program_node::CallInputError::WrongArity {
                expected: $crate::unpack_tensor_args!(@count $($x),+),
                actual: $args.len(),
            }
            .into());
        };
    };
    (@count $x:ident) => { 1usize };
    (@count $x:ident, $($rest:ident),+) => { 1usize + $crate::unpack_tensor_args!(@count $($rest),+) };
}

/// Errors returned when a tree-shaped argument does not match [`ProgramNode::input_types`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CallInputError {
    #[error("missing required input {key:?}")]
    MissingInput { key: String },

    #[error("expected a leaf at {key:?}, found a branch")]
    ExpectedLeaf { key: String },

    #[error("unexpected dtype at {key:?}: expected {expected}, found {actual}")]
    UnexpectedDType {
        key: String,
        expected: String,
        actual: DType,
    },

    #[error("expected {expected} total inputs, got {actual}")]
    WrongArity { expected: usize, actual: usize },
}

impl From<TreeMatchError> for CallInputError {
    fn from(e: TreeMatchError) -> Self {
        match e {
            TreeMatchError::MissingPath { path } => Self::MissingInput { key: path },
            TreeMatchError::ExpectedLeaf { path } => Self::ExpectedLeaf { key: path },
        }
    }
}

/// Returned by implementations with a missing call implementation when called.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("node {0:?} does not implement call()")]
pub struct MissingCallError(pub String);

impl MissingCallError {
    /// Construct a new [`MissingCallError`] tagged with the node's full name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Errors returned by [`ProgramNodeExt::call`].
#[derive(Debug, Error)]
pub enum CallError<E> {
    /// The input tree did not match the contract declared by `input_types()`.
    #[error(transparent)]
    Input(CallInputError),
    /// The node's [`ProgramNode::call_flat`] returned an error.
    #[error(transparent)]
    Call(E),
    /// The node's [`ProgramNode::call_flat`] returned a vector whose length
    /// did not match the leaf count of `output_types()`.
    #[error("call_flat returned {actual} outputs, expected {expected}")]
    OutputArityMismatch { expected: usize, actual: usize },
}

impl<E> From<ArityMismatch> for CallError<E> {
    fn from(e: ArityMismatch) -> Self {
        Self::OutputArityMismatch {
            expected: e.expected,
            actual: e.actual,
        }
    }
}

/// A node in a quantum program graph that transforms tensors.
pub trait ProgramNode {
    type CallError;

    /// The name of this program node.
    fn name(&self) -> &str;

    /// The namespace this program node belongs to.
    fn namespace(&self) -> &str;

    /// The namespace and name as one string.
    fn full_name(&self) -> String {
        format!("{}.{}", self.namespace(), self.name())
    }

    /// The inputs expected at call time.
    fn input_types(&self) -> &DataTree<TensorType>;

    /// The outputs promised on call return.
    fn output_types(&self) -> &DataTree<TensorType>;

    /// Whether this program node implements the call method.
    fn implements_call(&self) -> bool;

    /// The action of this program node with flattened I/O.
    ///
    /// `args` is in input-tree DFS leaf order matching `input_types()` and
    /// the returned vector is in output-tree DFS leaf order matching
    /// `output_types()`.
    ///
    /// # Panics
    ///
    /// Implementations are allowed to panic if `args.len()` does not equal
    /// the leaf count of `input_types()`; callers are responsible for upholding
    /// this invariant. On the other hand, implementations should raise a call
    /// error if they find tensors that they don't like.
    /// [`ProgramNodeExt::call`] and [`QuantumProgram::call_flat`] both do.
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError>;

    /// Resolve this node's output types with flattened I/O.
    ///
    /// `input_types` is in input-tree DFS leaf order matching [`ProgramNode::input_types`]
    /// and the returned vector is in output-tree DFS leaf order matching
    /// [`ProgramNode::output_types`]. This lets a node express shape/dtype dependence on
    /// its inputs (e.g. NumPy-style broadcasting, axis-removing reductions) as a real
    /// functional dependency, so mismatches can be caught before [`ProgramNode::call_flat`]
    /// ever runs, rather than only surfacing once real tensors are involved.
    ///
    /// The default implementation returns the frozen [`ProgramNode::output_types`] template
    /// unchanged, which is correct for nodes whose output types are independent of their
    /// input types (e.g. [`crate::Store`]).
    ///
    /// # Panics
    ///
    /// Implementations are allowed to panic if `input_types.len()` does not equal the leaf
    /// count of [`ProgramNode::input_types`]; callers are responsible for upholding this
    /// invariant. On the other hand, implementations should raise a call error if they find
    /// types that they don't like. [`ProgramNodeExt::resolve_types`] and
    /// [`QuantumProgram::resolve_types_flat`] both do.
    fn resolve_types_flat(
        &self,
        _input_types: &[TensorType],
    ) -> Result<Vec<TensorType>, Self::CallError> {
        Ok(self.output_types().iter_leaves().cloned().collect())
    }

    /// `Some(self)` if this node is a [`QuantumProgram`], else `None`.
    ///
    /// Nodes are stored type-erased behind a `dyn ProgramNode`, which has no `Any`
    /// supertrait and so cannot be downcast. This accessor exists so that callers can
    /// reach inside a nested program — most importantly the region nodes produced by
    /// [`QuantumProgram::into_regions`](crate::QuantumProgram::into_regions).
    ///
    /// Only [`QuantumProgram`] overrides this; everything else keeps the `None` default.
    fn as_quantum_program(&self) -> Option<&QuantumProgram> {
        None
    }
}

/// Extension with the wrapper over [`ProgramNode::call_flat`] whose I/O are data trees.
///
/// Provided via a blanket impl over every `T: ProgramNode` so that it cannot
/// be overridden in stable Rust.
pub trait ProgramNodeExt: ProgramNode {
    /// The action of this program node.
    fn call(
        &self,
        args: &DataTree<Tensor>,
    ) -> Result<DataTree<Tensor>, CallError<Self::CallError>> {
        let flat = self
            .input_types()
            .flatten_against(args)
            .map_err(|e| CallError::Input(e.into()))?;
        let out = self.call_flat(&flat).map_err(CallError::Call)?;
        self.output_types().unflatten(out).map_err(Into::into)
    }

    /// Resolve this node's output types given the types actually wired to its inputs.
    fn resolve_types(
        &self,
        input_types: &DataTree<TensorType>,
    ) -> Result<DataTree<TensorType>, CallError<Self::CallError>> {
        let flat = self
            .input_types()
            .flatten_against(input_types)
            .map_err(|e| CallError::Input(e.into()))?;
        let out = self.resolve_types_flat(&flat).map_err(CallError::Call)?;
        self.output_types().unflatten(out).map_err(Into::into)
    }
}

impl<T: ProgramNode + ?Sized> ProgramNodeExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DTypeLike, Dim};
    use std::sync::OnceLock;

    /// A node that doesn't override `resolve_types_flat`, to exercise the default.
    struct FixedNode;

    impl ProgramNode for FixedNode {
        type CallError = MissingCallError;
        fn name(&self) -> &'static str {
            "fixed"
        }
        fn namespace(&self) -> &'static str {
            "test"
        }
        fn input_types(&self) -> &DataTree<TensorType> {
            static LOCK: OnceLock<DataTree<TensorType>> = OnceLock::new();
            LOCK.get_or_init(|| {
                DataTree::new_leaf(TensorType {
                    dtype: DTypeLike::Concrete(DType::F64),
                    shape: vec![Dim::Fixed(3)],
                    broadcastable: false,
                })
            })
        }
        fn output_types(&self) -> &DataTree<TensorType> {
            static LOCK: OnceLock<DataTree<TensorType>> = OnceLock::new();
            LOCK.get_or_init(|| {
                DataTree::new_leaf(TensorType {
                    dtype: DTypeLike::Concrete(DType::Bit),
                    shape: vec![Dim::Fixed(5)],
                    broadcastable: false,
                })
            })
        }
        fn implements_call(&self) -> bool {
            false
        }
        fn call_flat(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, MissingCallError> {
            Err(MissingCallError::new(self.full_name()))
        }
    }

    #[test]
    fn test_default_resolve_types_flat_returns_output_types() {
        let node = FixedNode;
        let input = TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape: vec![Dim::Fixed(999)],
            broadcastable: false,
        };
        let result = node.resolve_types_flat(&[input]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(result[0].shape, vec![Dim::Fixed(5)]);
    }

    #[test]
    fn test_default_resolve_types_tree_wrapper() {
        let node = FixedNode;
        let input = DataTree::new_leaf(TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape: vec![Dim::Fixed(3)],
            broadcastable: false,
        });
        let result = node.resolve_types(&input).unwrap();
        let leaf = result.unwrap_leaf();
        assert!(matches!(leaf.dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(leaf.shape, vec![Dim::Fixed(5)]);
    }

    #[test]
    fn test_resolve_types_input_mismatch_errors() {
        let node = FixedNode;
        // A branch where a leaf is expected.
        let mut input = DataTree::new();
        input.insert_leaf(
            "a",
            TensorType {
                dtype: DTypeLike::Concrete(DType::F64),
                shape: vec![],
                broadcastable: false,
            },
        );
        let err = node.resolve_types(&input).unwrap_err();
        assert!(matches!(
            err,
            CallError::Input(CallInputError::ExpectedLeaf { ref key }) if key.is_empty()
        ));
    }
}
