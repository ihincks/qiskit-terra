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

use crate::data_tree::DataTree;
use crate::program_node::{MissingCallError, ProgramNode};
use crate::tensor::{Tensor, TensorType};
use std::sync::LazyLock;

// An empty data tree is the input for all `Input` nodes.
static EMPTY_DATA_TREE: LazyLock<DataTree<TensorType>> = LazyLock::new(DataTree::new);

/// A source node representing a program input, with no inputs of its own.
///
/// `Input` takes no inputs and does not implement [`ProgramNode::call_flat`]/
/// [`ProgramNode::resolve_types_flat`] directly — its value is injected by the enclosing
/// [`crate::QuantumProgram`] engine (see the program-input handling in
/// `QuantumProgram::run_topo`). In a data-flow graph, `Input` nodes exist so that a
/// single declared program input can fan out to multiple consumer ports through the
/// ordinary (and already-tested) edge-based fan-out mechanism, the same way [`crate::Store`]
/// lets a constant fan out.
pub struct Input {
    /// Single-leaf output type: the declared input spec.
    output_types: DataTree<TensorType>,
}

impl Input {
    /// Construct a new `Input` node with the given declared type.
    pub fn new(spec: TensorType) -> Self {
        Self {
            output_types: DataTree::new_leaf(spec),
        }
    }
}

impl ProgramNode for Input {
    type CallError = MissingCallError;

    fn name(&self) -> &str {
        "input"
    }

    fn namespace(&self) -> &str {
        "qiskit"
    }

    fn input_types(&self) -> &DataTree<TensorType> {
        &EMPTY_DATA_TREE
    }

    fn output_types(&self) -> &DataTree<TensorType> {
        &self.output_types
    }

    fn implements_call(&self) -> bool {
        false
    }

    fn call_flat(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, MissingCallError> {
        Err(MissingCallError::new(self.full_name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, DTypeLike, Dim};

    fn spec(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape,
            broadcastable: true,
        }
    }

    #[test]
    fn test_input_output_types_is_spec() {
        let node = Input::new(spec(vec![Dim::Fixed(3)]));
        let DataTree::Leaf(leaf) = node.output_types() else {
            panic!("expected a leaf output type");
        };
        assert!(matches!(leaf.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(leaf.shape, vec![Dim::Fixed(3)]);
    }

    #[test]
    fn test_input_has_no_inputs() {
        let node = Input::new(spec(vec![]));
        assert!(node.input_types().is_empty());
    }

    #[test]
    fn test_input_does_not_implement_call() {
        let node = Input::new(spec(vec![]));
        assert!(!node.implements_call());
        assert!(node.call_flat(&[]).is_err());
    }

    #[test]
    fn test_input_default_resolve_types_flat_returns_spec() {
        let node = Input::new(spec(vec![Dim::Fixed(5)]));
        let resolved = node.resolve_types_flat(&[]).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(5)]);
    }
}
