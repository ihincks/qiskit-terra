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
use crate::tensor::{DType, DTypeLike, Dim, Tensor, TensorType};
use crate::unpack_tensor_args;
use hashbrown::{HashMap, HashSet};
use ndarray::{ArrayD, Axis, IxDyn};
use qiskit_circuit::parameter::parameter_expression::{ParameterError, ParameterExpression};
use qiskit_circuit::parameter::symbol_expr::{Symbol, Value};
use thiserror::Error;

/// Errors returned when constructing a [`ParameterExpressions`] node.
///
/// These concern the declared parameter list itself, so they are raised up-front by
/// [`ParameterExpressions::new`] rather than deferred to a call.
#[derive(Debug, Error)]
pub enum ParameterExpressionsBuildError {
    /// The same parameter was declared more than once, so its input column would be ambiguous.
    #[error("parameter {name:?} was declared more than once")]
    DuplicateParameter { name: String },
    /// Expression `index` references a parameter that was not declared, so no input column
    /// supplies its value.
    #[error("expression {index} references undeclared parameter {name:?}")]
    UndeclaredParameter { index: usize, name: String },
}

/// Errors returned by [`ParameterExpressions`]'s [`ProgramNode`] implementation.
#[derive(Debug, Error)]
pub enum ParameterExpressionsError {
    /// The input tree did not match the contract declared by `input_types`.
    #[error(transparent)]
    Input(#[from] CallInputError),
    /// The input tensor did not have dtype `F64`.
    #[error("expected an f64 tensor, got dtype={dtype:?}")]
    WrongDType { dtype: DTypeLike },
    /// The input tensor did not have the right trailing (parameter-count) axis.
    #[error("expected a trailing axis of size {expected} (got shape {shape:?})")]
    WrongShape { shape: Vec<Dim>, expected: usize },
    /// The input tensor had no axes at all (not even the parameter axis).
    #[error("input tensor must have at least one axis (the parameter axis)")]
    EmptyShape,
    /// Binding parameter values into expression `index` failed.
    #[error("failed to evaluate expression {index}: {source}")]
    Eval {
        index: usize,
        source: ParameterError,
    },
    /// Expression `index` evaluated to a non-real value.
    #[error("expression {index} evaluated to a non-real value: {value:?}")]
    NonReal { index: usize, value: Value },
}

/// A program node that numerically evaluates a fixed list of [`ParameterExpression`]s.
///
/// This is the performant, Rust-native counterpart to a "parameter expression table":
/// given a batch of concrete values for the distinct atomic parameters spanned by its
/// expressions, it evaluates every expression for every row of the batch.
///
/// # Inputs
///
/// A single broadcastable `F64` tensor of shape `[..., num_parameters]`: the trailing axis
/// carries a value for each of this node's declared parameters, in the order they were given
/// to [`ParameterExpressions::new`], and any leading axes form an opaque batch prefix (of any
/// rank, including zero) specifying how many parameter sets to evaluate.
///
/// # Outputs
///
/// A single broadcastable `F64` tensor of shape `[..., num_expressions]`, where `...` is
/// the input's batch prefix, unchanged, and the trailing axis holds the value of each
/// expression (in the order they were given to [`ParameterExpressions::new`]).
pub struct ParameterExpressions {
    expressions: Vec<ParameterExpression>,
    /// The atomic parameters read off the input's trailing axis, in that axis's order.
    parameters: Vec<Symbol>,
    input_types: DataTree<TensorType>,
    output_types: DataTree<TensorType>,
}

impl ParameterExpressions {
    /// Construct a new `ParameterExpressions` node evaluating `expressions` from input values
    /// given for `parameters`, in that order.
    ///
    /// `parameters` must declare every parameter referenced by `expressions`, but may also
    /// declare extras — e.g. a whole circuit's parameters when an individual expression uses
    /// only some of them — whose input values are then simply ignored.
    pub fn new(
        expressions: Vec<ParameterExpression>,
        parameters: Vec<Symbol>,
    ) -> Result<Self, ParameterExpressionsBuildError> {
        Self::validate(&expressions, &parameters)?;

        let input_types = DataTree::new_leaf(TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape: vec![Dim::Fixed(parameters.len())],
            broadcastable: true,
        });
        let output_types = DataTree::new_leaf(TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape: vec![Dim::Fixed(expressions.len())],
            broadcastable: true,
        });

        Ok(Self {
            expressions,
            parameters,
            input_types,
            output_types,
        })
    }

    /// Check that `parameters` declares each of its entries exactly once, and that it covers
    /// every parameter referenced by `expressions`.
    fn validate(
        expressions: &[ParameterExpression],
        parameters: &[Symbol],
    ) -> Result<(), ParameterExpressionsBuildError> {
        let mut declared: HashSet<&Symbol> = HashSet::with_capacity(parameters.len());
        for parameter in parameters {
            if !declared.insert(parameter) {
                return Err(ParameterExpressionsBuildError::DuplicateParameter {
                    name: parameter.fullname().into_owned(),
                });
            }
        }
        for (index, expression) in expressions.iter().enumerate() {
            for symbol in expression.iter_symbols() {
                if !declared.contains(symbol) {
                    return Err(ParameterExpressionsBuildError::UndeclaredParameter {
                        index,
                        name: symbol.fullname().into_owned(),
                    });
                }
            }
        }
        Ok(())
    }

    /// The expressions this node evaluates, in output order.
    pub fn expressions(&self) -> &[ParameterExpression] {
        &self.expressions
    }

    /// The declared atomic parameters, in the order input values are read for them.
    pub fn parameters(&self) -> &[Symbol] {
        &self.parameters
    }

    /// The number of declared parameters (the input tensor's trailing axis size).
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// The number of expressions (the output tensor's trailing axis size).
    pub fn num_expressions(&self) -> usize {
        self.expressions.len()
    }
}

impl ProgramNode for ParameterExpressions {
    type CallError = ParameterExpressionsError;

    fn name(&self) -> &str {
        "parameter_expressions"
    }

    fn namespace(&self) -> &str {
        "qiskit"
    }

    fn input_types(&self) -> &DataTree<TensorType> {
        &self.input_types
    }

    fn output_types(&self) -> &DataTree<TensorType> {
        &self.output_types
    }

    fn implements_call(&self) -> bool {
        true
    }

    /// Bind and evaluate every expression for every row of the input's batch prefix.
    ///
    /// For each row along the trailing (parameter) axis, the row's values are bound to
    /// [`ParameterExpressions::parameters`] (in order) and every expression is evaluated
    /// against that binding, producing one row of the output's trailing (expression) axis.
    /// Values bound for parameters an expression doesn't reference are ignored.
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let Tensor::F64(arr) = x else {
            return Err(ParameterExpressionsError::WrongDType {
                dtype: DTypeLike::Concrete(x.dtype()),
            });
        };

        let shape = arr.shape();
        let Some((&last, batch_prefix)) = shape.split_last() else {
            return Err(ParameterExpressionsError::EmptyShape);
        };
        let n = self.parameters.len();
        if last != n {
            return Err(ParameterExpressionsError::WrongShape {
                shape: shape.iter().map(|&d| Dim::Fixed(d)).collect(),
                expected: n,
            });
        }

        let mut out_shape = batch_prefix.to_vec();
        out_shape.push(self.expressions.len());
        let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));

        let axis = Axis(shape.len() - 1);
        for (in_lane, mut out_lane) in arr.lanes(axis).into_iter().zip(out.lanes_mut(axis)) {
            let map: HashMap<&Symbol, Value> = self
                .parameters
                .iter()
                .zip(in_lane.iter())
                .map(|(symbol, &value)| (symbol, Value::Real(value)))
                .collect();

            for (i, expr) in self.expressions.iter().enumerate() {
                let bound = expr
                    .bind(&map, true)
                    .map_err(|source| ParameterExpressionsError::Eval { index: i, source })?;
                let value = bound
                    .try_to_value(true)
                    .map_err(|source| ParameterExpressionsError::Eval { index: i, source })?;
                if !value.is_real() {
                    return Err(ParameterExpressionsError::NonReal { index: i, value });
                }
                out_lane[i] = value.as_real();
            }
        }

        Ok(vec![Tensor::F64(out.into_shared())])
    }

    /// Resolve the output shape from the input's batch prefix.
    ///
    /// `input_types[0].shape` is split into `(batch_prefix, [last])`: `last` must equal
    /// [`ParameterExpressions::num_parameters`] (permissively accepted if it's an unresolved
    /// [`Dim::Named`]), and `batch_prefix` — of any rank, including zero — is prepended
    /// unchanged onto the output shape `[num_expressions]`. This mirrors
    /// [`crate::ShotLoop::resolve_types_flat`].
    fn resolve_types_flat(
        &self,
        input_types: &[TensorType],
    ) -> Result<Vec<TensorType>, Self::CallError> {
        unpack_tensor_args!(input_types, [x]);
        if !matches!(x.dtype, DTypeLike::Concrete(DType::F64)) {
            return Err(ParameterExpressionsError::WrongDType {
                dtype: x.dtype.clone(),
            });
        }

        let Some((last, batch_prefix)) = x.shape.split_last() else {
            return Err(ParameterExpressionsError::EmptyShape);
        };

        let n = self.parameters.len();
        let last_ok = match last {
            Dim::Fixed(k) => *k == n,
            Dim::Named(_) => true,
        };
        if !last_ok {
            return Err(ParameterExpressionsError::WrongShape {
                shape: x.shape.clone(),
                expected: n,
            });
        }

        let mut shape = batch_prefix.to_vec();
        shape.push(Dim::Fixed(self.expressions.len()));
        Ok(vec![TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape,
            broadcastable: true,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_node::{CallError, ProgramNodeExt};

    /// A fresh standalone parameter, paired with the expression that is just that parameter.
    fn param(name: &str) -> (Symbol, ParameterExpression) {
        let symbol = Symbol::standalone(name.to_string(), None);
        (symbol.clone(), ParameterExpression::from_symbol(symbol))
    }

    /// Build a node, panicking if `parameters` doesn't declare `expressions`' parameters.
    fn node(
        expressions: Vec<ParameterExpression>,
        parameters: Vec<Symbol>,
    ) -> ParameterExpressions {
        ParameterExpressions::new(expressions, parameters).unwrap()
    }

    fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
        Tensor::F64(
            ArrayD::from_shape_vec(IxDyn(&shape), data)
                .unwrap()
                .into_shared(),
        )
    }

    fn f64_type(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape,
            broadcastable: true,
        }
    }

    #[test]
    fn test_name_and_namespace() {
        let node = node(vec![], vec![]);
        assert_eq!(node.name(), "parameter_expressions");
        assert_eq!(node.namespace(), "qiskit");
        assert_eq!(node.full_name(), "qiskit.parameter_expressions");
    }

    #[test]
    fn test_implements_call() {
        assert!(node(vec![], vec![]).implements_call());
    }

    #[test]
    fn test_declared_parameter_order_is_respected() {
        // Declaring [b, a] means the input's first column is b and its second is a, even
        // though the canonical sort of the two would be [a, b].
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a, b], vec![b_param, a_param]);

        assert_eq!(node.num_parameters(), 2);
        assert_eq!(node.num_expressions(), 2);
        let names: Vec<&str> = node.parameters().iter().map(|s| s.name()).collect();
        assert_eq!(names, vec!["b", "a"]);

        // b=10, a=20, so the expressions [a, b] evaluate to [20, 10].
        let result = node
            .call_flat(&[f64_tensor(vec![2], vec![10.0, 20.0])])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.as_slice().unwrap(), &[20.0, 10.0]);
    }

    #[test]
    fn test_superset_parameters_allowed() {
        // Declaring parameters the expression doesn't reference is fine: their input column
        // is simply ignored, and the input axis stays as wide as the declaration.
        let (a_param, _) = param("a");
        let (b_param, b) = param("b");
        let (c_param, _) = param("c");
        let node = node(vec![b], vec![a_param, b_param, c_param]);
        assert_eq!(node.num_parameters(), 3);

        let result = node
            .call_flat(&[f64_tensor(vec![3], vec![1.0, 2.0, 3.0])])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn test_undeclared_parameter_errors() {
        let (a_param, a) = param("a");
        let (_, b) = param("b");
        let Err(err) = ParameterExpressions::new(vec![a, b], vec![a_param]) else {
            panic!("expected an undeclared-parameter error");
        };
        assert!(matches!(
            err,
            ParameterExpressionsBuildError::UndeclaredParameter { index: 1, ref name }
                if name == "b"
        ));
    }

    #[test]
    fn test_duplicate_parameter_errors() {
        let (a_param, a) = param("a");
        let Err(err) = ParameterExpressions::new(vec![a], vec![a_param.clone(), a_param]) else {
            panic!("expected a duplicate-parameter error");
        };
        assert!(matches!(
            err,
            ParameterExpressionsBuildError::DuplicateParameter { ref name } if name == "a"
        ));
    }

    #[test]
    fn test_input_output_types_shape() {
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(
            vec![a.clone(), b.clone(), a.add(&b).unwrap()],
            vec![a_param, b_param],
        );

        let DataTree::Leaf(input_ty) = node.input_types() else {
            panic!("expected a leaf input type");
        };
        assert!(matches!(input_ty.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(input_ty.shape, vec![Dim::Fixed(2)]);

        let DataTree::Leaf(output_ty) = node.output_types() else {
            panic!("expected a leaf output type");
        };
        assert_eq!(output_ty.shape, vec![Dim::Fixed(3)]);
    }

    #[test]
    fn test_no_expressions_has_empty_types() {
        let node = node(vec![], vec![]);
        assert_eq!(node.num_parameters(), 0);
        assert_eq!(node.num_expressions(), 0);
        let DataTree::Leaf(input_ty) = node.input_types() else {
            panic!("expected a leaf input type");
        };
        assert_eq!(input_ty.shape, vec![Dim::Fixed(0)]);
    }

    // -----------------------------------------------------------------------
    // call_flat
    // -----------------------------------------------------------------------

    #[test]
    fn test_call_flat_unbatched() {
        // a + b, a * b, with a=2, b=3 -> [5, 6].
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(
            vec![a.clone().add(&b).unwrap(), a.clone().mul(&b).unwrap()],
            vec![a_param, b_param],
        );

        let input = f64_tensor(vec![2], vec![2.0, 3.0]);
        let result = node.call_flat(&[input]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.as_slice().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_call_flat_batched() {
        // Same expressions as above, but with a batch prefix of 3 rows.
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a.add(&b).unwrap()], vec![a_param, b_param]);

        // rows: (1, 1), (2, 3), (10, -1) -> sums [2, 5, 9]
        let input = f64_tensor(vec![3, 2], vec![1.0, 1.0, 2.0, 3.0, 10.0, -1.0]);
        let result = node.call_flat(&[input]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.shape(), &[3, 1]);
        assert_eq!(arr.as_slice().unwrap(), &[2.0, 5.0, 9.0]);
    }

    #[test]
    fn test_call_flat_multi_batch_dim() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);

        // shape [2, 2, 1] -> output shape [2, 2, 1], values passed through unchanged.
        let input = f64_tensor(vec![2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);
        let result = node.call_flat(&[input]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.shape(), &[2, 2, 1]);
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_call_flat_no_parameters() {
        // A constant expression, with a batch of 4 rows and a zero-length parameter axis.
        let node = node(vec![ParameterExpression::from_f64(7.0)], vec![]);
        assert_eq!(node.num_parameters(), 0);

        let input = f64_tensor(vec![4, 0], vec![]);
        let result = node.call_flat(&[input]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.shape(), &[4, 1]);
        assert_eq!(arr.as_slice().unwrap(), &[7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_call_flat_no_expressions() {
        let node = node(vec![], vec![]);
        let input = f64_tensor(vec![3, 0], vec![]);
        let result = node.call_flat(&[input]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.shape(), &[3, 0]);
    }

    #[test]
    fn test_call_flat_wrong_dtype_errors() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let input = Tensor::from([1_i32]);
        let err = node.call_flat(&[input]).unwrap_err();
        assert!(matches!(
            err,
            ParameterExpressionsError::WrongDType {
                dtype: DTypeLike::Concrete(DType::I32)
            }
        ));
    }

    #[test]
    fn test_call_flat_wrong_trailing_dim_errors() {
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a, b], vec![a_param, b_param]);
        let input = f64_tensor(vec![1], vec![1.0]);
        let err = node.call_flat(&[input]).unwrap_err();
        assert!(matches!(
            err,
            ParameterExpressionsError::WrongShape { expected: 2, .. }
        ));
    }

    #[test]
    fn test_call_flat_empty_shape_errors() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let input = Tensor::F64(
            ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0])
                .unwrap()
                .into_shared(),
        );
        let err = node.call_flat(&[input]).unwrap_err();
        assert!(matches!(err, ParameterExpressionsError::EmptyShape));
    }

    #[test]
    fn test_call_flat_end_to_end_via_call() {
        let (a_param, a) = param("a");
        let node = node(vec![a.clone(), a.mul(&a).unwrap()], vec![a_param]);
        let tree = DataTree::new_leaf(f64_tensor(vec![1], vec![4.0]));
        let result = node.call(&tree).unwrap();
        let Tensor::F64(arr) = result.unwrap_leaf() else {
            panic!("expected F64 output");
        };
        assert_eq!(arr.as_slice().unwrap(), &[4.0, 16.0]);
    }

    // -----------------------------------------------------------------------
    // resolve_types_flat
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_types_flat_unbatched() {
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a, b], vec![a_param, b_param]);
        let resolved = node
            .resolve_types_flat(&[f64_type(vec![Dim::Fixed(2)])])
            .unwrap();
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(2)]);
    }

    #[test]
    fn test_resolve_types_flat_batch_prefix() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let resolved = node
            .resolve_types_flat(&[f64_type(vec![
                Dim::Fixed(10),
                Dim::Fixed(20),
                Dim::Fixed(1),
            ])])
            .unwrap();
        assert_eq!(
            resolved[0].shape,
            vec![Dim::Fixed(10), Dim::Fixed(20), Dim::Fixed(1)]
        );
    }

    #[test]
    fn test_resolve_types_flat_named_trailing_dim_accepted() {
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a, b], vec![a_param, b_param]);
        let resolved = node
            .resolve_types_flat(&[f64_type(vec![Dim::Fixed(5), Dim::Named("n".into())])])
            .unwrap();
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(5), Dim::Fixed(2)]);
    }

    #[test]
    fn test_resolve_types_flat_wrong_trailing_dim_errors() {
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a, b], vec![a_param, b_param]);
        let err = node
            .resolve_types_flat(&[f64_type(vec![Dim::Fixed(3)])])
            .unwrap_err();
        assert!(matches!(
            err,
            ParameterExpressionsError::WrongShape { expected: 2, .. }
        ));
    }

    #[test]
    fn test_resolve_types_flat_wrong_dtype_errors() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let bit_type = TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape: vec![Dim::Fixed(1)],
            broadcastable: true,
        };
        let err = node.resolve_types_flat(&[bit_type]).unwrap_err();
        assert!(matches!(err, ParameterExpressionsError::WrongDType { .. }));
    }

    #[test]
    fn test_resolve_types_flat_empty_shape_errors() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let err = node.resolve_types_flat(&[f64_type(vec![])]).unwrap_err();
        assert!(matches!(err, ParameterExpressionsError::EmptyShape));
    }

    #[test]
    fn test_resolve_types_flat_matches_call_flat_shape() {
        let (a_param, a) = param("a");
        let (b_param, b) = param("b");
        let node = node(vec![a.add(&b).unwrap()], vec![a_param, b_param]);

        let input = f64_tensor(vec![5, 2], vec![0.0; 10]);
        let called = node.call_flat(&[input]).unwrap();
        let resolved = node
            .resolve_types_flat(&[f64_type(vec![Dim::Fixed(5), Dim::Fixed(2)])])
            .unwrap();
        assert_eq!(called[0].shape(), &[5, 1]);
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(5), Dim::Fixed(1)]);
    }

    #[test]
    fn test_call_wrong_arity_errors() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let err = node
            .call_flat(&[
                f64_tensor(vec![1], vec![1.0]),
                f64_tensor(vec![1], vec![1.0]),
            ])
            .unwrap_err();
        assert!(matches!(
            err,
            ParameterExpressionsError::Input(CallInputError::WrongArity {
                expected: 1,
                actual: 2,
            })
        ));
    }

    #[test]
    fn test_call_branch_where_leaf_expected_errors() {
        let (a_param, a) = param("a");
        let node = node(vec![a], vec![a_param]);
        let mut tree = DataTree::new();
        tree.insert_leaf("x", f64_tensor(vec![1], vec![1.0]));
        let err = node.call(&tree).unwrap_err();
        assert!(matches!(
            err,
            CallError::<ParameterExpressionsError>::Input(CallInputError::ExpectedLeaf {
                ref key,
            }) if key.is_empty()
        ));
    }
}
