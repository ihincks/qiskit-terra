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
use crate::program_node::{CallInputError, ProgramNode};
use crate::tensor::{DType, DTypeLike, Tensor, TensorType, broadcast_dims, broadcast_shape};
use crate::unpack_tensor_args;
use ndarray::Axis;
use std::sync::LazyLock;

/// Shared input type spec for binary bitwise nodes
static INPUT_TYPES: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    let mut types = DataTree::with_capacity(2);
    types.insert_leaf(
        "x",
        TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape: vec![],
            broadcastable: true,
        },
    );
    types.insert_leaf(
        "y",
        TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape: vec![],
            broadcastable: true,
        },
    );
    types
});

/// A single broadcastable `Bit` leaf — used for unary inputs and all bitwise outputs.
static LEAF_TYPE: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    DataTree::new_leaf(TensorType {
        dtype: DTypeLike::Concrete(DType::Bit),
        shape: vec![],
        broadcastable: true,
    })
});

/// Construct an `UnexpectedDType` error for a value that did not match
/// the schema's required dtype.
fn unexpected_dtype(key: &str, actual: DType) -> CallInputError {
    CallInputError::UnexpectedDType {
        key: key.into(),
        expected: DType::Bit.to_string(),
        actual,
    }
}

/// Validate that a declared (possibly symbolic) dtype is `Bit`, deferring to `call_flat`
/// when the dtype is not yet resolved to a concrete type.
fn check_bit_dtype(key: &str, dtype: &DTypeLike) -> Result<(), CallInputError> {
    match dtype {
        DTypeLike::Concrete(d) if *d != DType::Bit => Err(unexpected_dtype(key, *d)),
        _ => Ok(()),
    }
}

/// Generate a [`ProgramNode`] struct for an elementwise binary bitwise operation on `Bit` tensors.
macro_rules! bitwise_binary_node {
    ($name:ident, $node_name:literal, $call_fn:expr) => {
        #[doc = concat!("Elementwise `", $node_name, "` of two broadcastable `Bit` tensors.")]
        pub struct $name;

        impl ProgramNode for $name {
            type CallError = super::MathNodeError;

            fn name(&self) -> &'static str {
                $node_name
            }
            fn namespace(&self) -> &'static str {
                "qiskit"
            }
            fn input_types(&self) -> &DataTree<TensorType> {
                &INPUT_TYPES
            }
            fn output_types(&self) -> &DataTree<TensorType> {
                &LEAF_TYPE
            }
            fn implements_call(&self) -> bool {
                true
            }
            fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
                unpack_tensor_args!(args, [x, y]);
                let Tensor::Bit(x_arr) = x else {
                    return Err(unexpected_dtype("x", x.dtype()).into());
                };
                let Tensor::Bit(y_arr) = y else {
                    return Err(unexpected_dtype("y", y.dtype()).into());
                };
                broadcast_shape(x_arr.shape(), y_arr.shape())?;
                Ok(vec![Tensor::Bit($call_fn(x_arr, y_arr).into_shared())])
            }
            fn resolve_types_flat(
                &self,
                input_types: &[TensorType],
            ) -> Result<Vec<TensorType>, Self::CallError> {
                unpack_tensor_args!(input_types, [x, y]);
                check_bit_dtype("x", &x.dtype)?;
                check_bit_dtype("y", &y.dtype)?;
                let shape = broadcast_dims(&x.shape, &y.shape)?;
                Ok(vec![TensorType {
                    dtype: DTypeLike::Concrete(DType::Bit),
                    shape,
                    broadcastable: true,
                }])
            }
        }
    };
}

bitwise_binary_node!(BitwiseAnd, "bitwise_and", |x, y| x & y);
bitwise_binary_node!(BitwiseOr, "bitwise_or", |x, y| x | y);
bitwise_binary_node!(BitwiseXor, "bitwise_xor", |x, y| x ^ y);

/// Elementwise bitwise NOT of a broadcastable `Bit` tensor.
pub struct BitwiseNot;

impl ProgramNode for BitwiseNot {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "bitwise_not"
    }
    fn namespace(&self) -> &'static str {
        "qiskit"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let Tensor::Bit(arr) = x else {
            return Err(unexpected_dtype("", x.dtype()).into());
        };
        Ok(vec![Tensor::Bit(arr.mapv(|b| b ^ 1).into_shared())])
    }
    fn resolve_types_flat(
        &self,
        input_types: &[TensorType],
    ) -> Result<Vec<TensorType>, Self::CallError> {
        unpack_tensor_args!(input_types, [x]);
        check_bit_dtype("", &x.dtype)?;
        Ok(vec![x.clone()])
    }
}

/// XOR-reduction of a `Bit` tensor along a specified axis, removing that axis.
///
/// The parity of a sequence of bits is 1 if an odd number of bits are 1, and 0 otherwise,
/// which is equivalent to XOR-folding the sequence. The output has one fewer dimension than
/// the input, with the reduction axis removed.
pub struct Parity {
    axis: usize,
}

impl Parity {
    /// Construct a `Parity` node that reduces along `axis`.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl ProgramNode for Parity {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "parity"
    }
    fn namespace(&self) -> &'static str {
        "qiskit"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        super::check_axis(self.axis, x.shape().len())?;
        let Tensor::Bit(arr) = x else {
            return Err(unexpected_dtype("", x.dtype()).into());
        };
        Ok(vec![Tensor::Bit(
            arr.fold_axis(Axis(self.axis), 0u8, |&acc, &b| acc ^ b)
                .into_shared(),
        )])
    }
    fn resolve_types_flat(
        &self,
        input_types: &[TensorType],
    ) -> Result<Vec<TensorType>, Self::CallError> {
        unpack_tensor_args!(input_types, [x]);
        super::check_axis(self.axis, x.shape.len())?;
        check_bit_dtype("", &x.dtype)?;
        let mut shape = x.shape.clone();
        shape.remove(self.axis);
        Ok(vec![TensorType {
            dtype: x.dtype.clone(),
            shape,
            broadcastable: x.broadcastable,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_nodes::MathNodeError;
    use crate::program_node::{CallError, CallInputError, ProgramNodeExt};
    use ndarray::{arr1, arr2};

    fn bit(data: &[u8]) -> Tensor {
        Tensor::Bit(arr1(data).into_dyn().into_shared())
    }

    #[test]
    fn test_bitwise_and() {
        let result = BitwiseAnd
            .call_flat(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 0, 0, 1]);
    }

    #[test]
    fn test_bitwise_or() {
        let result = BitwiseOr
            .call_flat(&[bit(&[1, 0, 1, 0]), bit(&[0, 1, 0, 1])])
            .unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_bitwise_xor() {
        let result = BitwiseXor
            .call_flat(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[0, 1, 1, 0]);
    }

    #[test]
    fn test_bitwise_and_broadcasts() {
        // shape [3] & shape [1] -> shape [3]
        let result = BitwiseAnd.call_flat(&[bit(&[1, 0, 1]), bit(&[1])]).unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 0, 1]);
    }

    #[test]
    fn test_bitwise_not() {
        let result = BitwiseNot.call_flat(&[bit(&[1, 0, 1, 0])]).unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[0, 1, 0, 1]);
    }

    #[test]
    fn test_parity_axis0() {
        // [[1,0,1],[0,1,1],[0,0,0]] axis 0 → [1, 1, 0]
        let x = Tensor::Bit(
            arr2(&[[1u8, 0, 1], [0, 1, 1], [0, 0, 0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Parity::new(0).call_flat(&[x]).unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 1, 0]);
    }

    #[test]
    fn test_bitwise_and_wrong_dtype_errors() {
        let err = BitwiseAnd
            .call_flat(&[Tensor::from([1.0_f64]), bit(&[1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::UnexpectedDType {
                key: "x".to_string(),
                expected: "Bit".to_string(),
                actual: DType::F64,
            })
        );
    }

    #[test]
    fn test_bitwise_and_wrong_arity_errors() {
        let err = BitwiseAnd.call_flat(&[bit(&[1, 0])]).unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::WrongArity {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn test_bitwise_not_wrong_arity_errors() {
        let err = BitwiseNot
            .call_flat(&[bit(&[1, 0]), bit(&[0, 1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::WrongArity {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn test_bitwise_and_shape_mismatch_errors() {
        let err = BitwiseAnd
            .call_flat(&[bit(&[1, 0, 1]), bit(&[1, 0, 1, 1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Tensor(crate::tensor::TensorError::ShapeMismatch {
                lhs: vec![3],
                rhs: vec![4],
            })
        );
    }

    #[test]
    fn test_call_branch_where_leaf_expected_errors() {
        let mut tree = DataTree::new();
        tree.insert_leaf("x", bit(&[1, 0]));
        let err = BitwiseNot.call(&tree).unwrap_err();
        assert!(matches!(
            err,
            CallError::<MathNodeError>::Input(CallInputError::ExpectedLeaf {
                ref key,
            }) if key.is_empty()
        ));
    }

    #[test]
    fn test_bitwise_and_call_end_to_end() {
        let mut tree = DataTree::new();
        tree.insert_leaf("x", bit(&[1, 0, 1, 1]));
        tree.insert_leaf("y", bit(&[1, 1, 0, 1]));
        let result = BitwiseAnd.call(&tree).unwrap();
        let Tensor::Bit(arr) = result.unwrap_leaf() else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 0, 0, 1]);
    }

    #[test]
    fn test_parity_axis_out_of_bounds_errors() {
        let err = Parity::new(1).call_flat(&[bit(&[1, 0, 1])]).unwrap_err();
        assert_eq!(err, MathNodeError::InvalidAxis { axis: 1, ndim: 1 });
    }

    // -----------------------------------------------------------------------
    // resolve_types_flat
    // -----------------------------------------------------------------------

    use crate::tensor::Dim;

    fn concrete_bit(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape,
            broadcastable: true,
        }
    }

    #[test]
    fn test_bitwise_and_resolve_types_flat_matches_call_flat() {
        let called = BitwiseAnd
            .call_flat(&[bit(&[1, 0, 1]), bit(&[1, 1, 0])])
            .unwrap();
        let resolved = BitwiseAnd
            .resolve_types_flat(&[
                concrete_bit(vec![Dim::Fixed(3)]),
                concrete_bit(vec![Dim::Fixed(3)]),
            ])
            .unwrap();
        assert_eq!(resolved.len(), 1);
        assert!(matches!(resolved[0].dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(3)]);
        assert_eq!(called[0].shape(), &[3]);
    }

    #[test]
    fn test_bitwise_and_resolve_types_flat_broadcasts() {
        let resolved = BitwiseAnd
            .resolve_types_flat(&[
                concrete_bit(vec![Dim::Fixed(3)]),
                concrete_bit(vec![Dim::Fixed(1)]),
            ])
            .unwrap();
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(3)]);
    }

    #[test]
    fn test_bitwise_and_resolve_types_flat_wrong_dtype_errors() {
        let err = BitwiseAnd
            .resolve_types_flat(&[
                TensorType {
                    dtype: DTypeLike::Concrete(DType::F64),
                    shape: vec![],
                    broadcastable: true,
                },
                concrete_bit(vec![]),
            ])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::UnexpectedDType {
                key: "x".to_string(),
                expected: "Bit".to_string(),
                actual: DType::F64,
            })
        );
    }

    #[test]
    fn test_bitwise_and_resolve_types_flat_incompatible_shapes_errors() {
        let err = BitwiseAnd
            .resolve_types_flat(&[
                concrete_bit(vec![Dim::Fixed(3)]),
                concrete_bit(vec![Dim::Fixed(4)]),
            ])
            .unwrap_err();
        assert!(matches!(err, MathNodeError::Tensor(_)));
    }

    #[test]
    fn test_bitwise_and_resolve_types_flat_permits_symbolic_dtype() {
        // A symbolic (unresolved) dtype is not checked against `Bit` here — that
        // validation is deferred to `call_flat` once the dtype is concrete.
        let resolved = BitwiseAnd
            .resolve_types_flat(&[
                TensorType {
                    dtype: DTypeLike::Var("x".into()),
                    shape: vec![Dim::Fixed(3)],
                    broadcastable: true,
                },
                concrete_bit(vec![Dim::Fixed(3)]),
            ])
            .unwrap();
        assert!(matches!(resolved[0].dtype, DTypeLike::Concrete(DType::Bit)));
    }

    #[test]
    fn test_bitwise_not_resolve_types_flat_passes_through() {
        let input = concrete_bit(vec![Dim::Fixed(4)]);
        let resolved = BitwiseNot
            .resolve_types_flat(std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(resolved.len(), 1);
        assert!(matches!(resolved[0].dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(resolved[0].shape, input.shape);
        assert_eq!(resolved[0].broadcastable, input.broadcastable);
    }

    #[test]
    fn test_bitwise_not_resolve_types_flat_wrong_dtype_errors() {
        let err = BitwiseNot
            .resolve_types_flat(&[TensorType {
                dtype: DTypeLike::Concrete(DType::F64),
                shape: vec![],
                broadcastable: true,
            }])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::UnexpectedDType {
                key: "".to_string(),
                expected: "Bit".to_string(),
                actual: DType::F64,
            })
        );
    }

    #[test]
    fn test_parity_resolve_types_flat_removes_axis() {
        let called = Parity::new(0)
            .call_flat(&[Tensor::Bit(
                arr2(&[[1u8, 0, 1], [0, 1, 1]]).into_dyn().into_shared(),
            )])
            .unwrap();
        let resolved = Parity::new(0)
            .resolve_types_flat(&[concrete_bit(vec![Dim::Fixed(2), Dim::Fixed(3)])])
            .unwrap();
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(3)]);
        assert_eq!(called[0].shape(), &[3]);
    }

    #[test]
    fn test_parity_resolve_types_flat_axis_out_of_bounds_errors() {
        let err = Parity::new(1)
            .resolve_types_flat(&[concrete_bit(vec![Dim::Fixed(3)])])
            .unwrap_err();
        assert_eq!(err, MathNodeError::InvalidAxis { axis: 1, ndim: 1 });
    }
}
