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
use crate::program_node::{CallInputError, MissingCallError, ProgramNode};
use crate::tensor::{DType, DTypeLike, Dim, Tensor, TensorType};
use qiskit_circuit::circuit_data::CircuitData;
use thiserror::Error;

/// Errors returned by [`ShotLoop`]'s [`ProgramNode`] implementation.
#[derive(Debug, Clone, Error)]
pub enum ShotLoopError {
    /// The input tree did not match the contract declared by `input_types`.
    #[error(transparent)]
    Input(#[from] CallInputError),
    /// [`ShotLoop::call_flat`] was invoked; `ShotLoop` never implements `call`.
    #[error(transparent)]
    MissingCall(#[from] MissingCallError),
    /// A circuit's params tensor did not have dtype `F64`.
    #[error("params[{circuit}] must be an f64 tensor, got dtype={dtype:?}")]
    WrongParamDType { circuit: usize, dtype: DTypeLike },
    /// A circuit's params tensor did not have the right trailing (parameter-count) axis.
    #[error("params[{circuit}] must have a trailing axis of size {expected} (got shape {shape:?})")]
    WrongParamShape {
        circuit: usize,
        shape: Vec<Dim>,
        expected: usize,
    },
    /// A circuit's params tensor had no axes at all (not even the parameter axis).
    #[error("params[{circuit}] must have at least one axis (the parameter axis)")]
    EmptyParamShape { circuit: usize },
}

/// A program node that runs a fixed list of circuits, each for the same number of shots.
///
/// `ShotLoop` is the canonical "remote" node — it has no local execution path
/// and its [`ProgramNode::call_flat`] always returns [`MissingCallError`]. A
/// backend is expected to dispatch it to hardware (or a simulator) and produce
/// the declared output bitstrings.
///
/// # Inputs
///
/// One leaf per circuit, list-indexed in the order the circuits were given.
/// The leaf for circuit `i` is a broadcastable `F64` tensor of shape
/// `[..., num_parameters_i]`: the trailing axis carries that circuit's parameter
/// values, and any leading axes form an opaque batch prefix (of any rank, including
/// zero) specifying how many parameter sets to run. See [`ProgramNode::resolve_types_flat`]
/// for how this prefix propagates to the outputs.
///
/// # Outputs
///
/// One branch per circuit, list-indexed. Each branch contains one leaf per
/// classical register, keyed by the register's name. The leaf is a
/// broadcastable `Bit` tensor of shape `[..., shots, num_bits]`, where `...` is
/// that circuit's input batch prefix, unchanged.
pub struct ShotLoop {
    circuits: Vec<CircuitData>,
    shots: usize,
    input_types: DataTree<TensorType>,
    output_types: DataTree<TensorType>,
}

impl ShotLoop {
    /// Construct a new `ShotLoop` for the given `circuits` and `shots`.
    pub fn new(circuits: Vec<CircuitData>, shots: usize) -> Self {
        let mut input_types = DataTree::with_capacity(circuits.len());
        let mut output_types = DataTree::with_capacity(circuits.len());

        for circuit in &circuits {
            input_types.push_leaf(TensorType {
                dtype: DTypeLike::Concrete(DType::F64),
                shape: vec![Dim::Fixed(circuit.num_parameters())],
                broadcastable: true,
            });

            let cregs = circuit.cregs();
            let mut branch = DataTree::with_capacity(cregs.len());
            for creg in cregs {
                branch.insert_leaf(
                    creg.name(),
                    TensorType {
                        dtype: DTypeLike::Concrete(DType::Bit),
                        shape: vec![Dim::Fixed(shots), Dim::Fixed(creg.len())],
                        broadcastable: true,
                    },
                );
            }
            output_types.push_branch(branch);
        }

        Self {
            circuits,
            shots,
            input_types,
            output_types,
        }
    }

    /// The circuits this `ShotLoop` will run.
    pub fn circuits(&self) -> &[CircuitData] {
        &self.circuits
    }

    /// The number of shots each circuit will be run for.
    pub fn shots(&self) -> usize {
        self.shots
    }
}

impl ProgramNode for ShotLoop {
    type CallError = ShotLoopError;

    fn name(&self) -> &str {
        "shot_loop"
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
        false
    }

    fn call_flat(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        Err(MissingCallError::new(self.full_name()).into())
    }

    /// Resolve each circuit's output register shapes from its params shape.
    ///
    /// For circuit `i`, `input_types[i].shape` is split into `(batch_prefix, [last])`:
    /// `last` must equal `circuit.num_parameters()` (permissively accepted if it's an
    /// unresolved [`Dim::Named`]), and `batch_prefix` — of any rank, including zero — is
    /// prepended unchanged onto every one of that circuit's output register shapes
    /// (which are otherwise `[shots, num_bits]`). This makes batching rank-agnostic:
    /// `f64[n]`, `f64[10, n]`, and `f64[10, 20, 30, n]` all resolve via the same code path.
    fn resolve_types_flat(
        &self,
        input_types: &[TensorType],
    ) -> Result<Vec<TensorType>, Self::CallError> {
        let mut outputs = Vec::new();

        for (i, (circuit, input_ty)) in self.circuits.iter().zip(input_types).enumerate() {
            if !matches!(input_ty.dtype, DTypeLike::Concrete(DType::F64)) {
                return Err(ShotLoopError::WrongParamDType {
                    circuit: i,
                    dtype: input_ty.dtype.clone(),
                });
            }

            let Some((last, batch_prefix)) = input_ty.shape.split_last() else {
                return Err(ShotLoopError::EmptyParamShape { circuit: i });
            };

            let expected = circuit.num_parameters();
            let last_ok = match last {
                Dim::Fixed(n) => *n == expected,
                Dim::Named(_) => true,
            };
            if !last_ok {
                return Err(ShotLoopError::WrongParamShape {
                    circuit: i,
                    shape: input_ty.shape.clone(),
                    expected,
                });
            }

            for creg in circuit.cregs() {
                let mut shape = batch_prefix.to_vec();
                shape.push(Dim::Fixed(self.shots));
                shape.push(Dim::Fixed(creg.len()));
                outputs.push(TensorType {
                    dtype: DTypeLike::Concrete(DType::Bit),
                    shape,
                    broadcastable: true,
                });
            }
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::{ClassicalRegister, ShareableQubit};
    use qiskit_circuit::operations::{Param, StandardGate};
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;
    use std::sync::Arc;

    /// Build a `CircuitData` with the given classical registers (and no
    /// quantum bits or instructions). Sufficient for exercising the layout of
    /// `ShotLoop`'s input and output type trees.
    fn circuit_with_cregs(cregs: Vec<ClassicalRegister>) -> CircuitData {
        let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
        for creg in cregs {
            circuit.add_creg(creg, true).unwrap();
        }
        circuit
    }

    /// Build a `CircuitData` with `num_params` distinct free parameters (each applied
    /// via a single-qubit `RX` gate on a fresh anonymous qubit) plus the given classical
    /// registers. Sufficient for exercising `resolve_types_flat`'s parameter-count check.
    fn circuit_with_params(num_params: usize, cregs: Vec<ClassicalRegister>) -> CircuitData {
        let qubits = vec![ShareableQubit::new_anonymous()];
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        for i in 0..num_params {
            let param = Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone(format!("p{i}"), None),
            )));
            circuit
                .push_standard_gate(StandardGate::RX, &[param], &[Qubit(0)])
                .unwrap();
        }
        for creg in cregs {
            circuit.add_creg(creg, true).unwrap();
        }
        circuit
    }

    #[test]
    fn test_name_and_namespace() {
        let sl = ShotLoop::new(vec![], 100);
        assert_eq!(sl.name(), "shot_loop");
        assert_eq!(sl.namespace(), "qiskit");
        assert_eq!(sl.full_name(), "qiskit.shot_loop");
    }

    #[test]
    fn test_does_not_implement_call() {
        let sl = ShotLoop::new(vec![], 100);
        assert!(!sl.implements_call());
    }

    #[test]
    fn test_call_returns_missing_call_error() {
        let sl = ShotLoop::new(vec![], 100);
        let err = sl.call_flat(&[]).unwrap_err();
        assert!(matches!(
            err,
            ShotLoopError::MissingCall(ref e) if *e == MissingCallError::new("qiskit.shot_loop")
        ));
    }

    #[test]
    fn test_empty_input_and_output_types() {
        // No circuits → empty input and output type trees.
        let sl = ShotLoop::new(vec![], 100);
        assert!(sl.input_types().is_empty());
        assert!(sl.output_types().is_empty());
    }

    #[test]
    fn test_input_types_shape_zero_params() {
        // A non-parametric circuit has shape [Fixed(0)] for its parameter input.
        let sl = ShotLoop::new(vec![circuit_with_cregs(vec![])], 100);

        assert_eq!(sl.input_types().len(), 1);
        let DataTree::Leaf(tt) = sl.input_types().get(0).unwrap() else {
            panic!("expected a leaf for circuit 0's parameters");
        };
        assert!(matches!(tt.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(tt.shape, vec![Dim::Fixed(0)]);
        assert!(tt.broadcastable);
    }

    #[test]
    fn test_output_types_register_layout() {
        // One circuit with two classical registers of different sizes.
        let circuit = circuit_with_cregs(vec![
            ClassicalRegister::new_owning("c", 3),
            ClassicalRegister::new_owning("meas", 5),
        ]);
        let sl = ShotLoop::new(vec![circuit], 1024);

        // The output is a branch keyed by circuit index, each entry is itself
        // a branch keyed by register name.
        assert_eq!(sl.output_types().len(), 1);
        let circ_branch = sl.output_types().get(0).unwrap();

        let DataTree::Leaf(c_tt) = circ_branch.get_by_str_key("c").unwrap() else {
            panic!("expected a leaf at register 'c'");
        };
        assert!(matches!(c_tt.dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(c_tt.shape, vec![Dim::Fixed(1024), Dim::Fixed(3)]);
        assert!(c_tt.broadcastable);

        let DataTree::Leaf(meas_tt) = circ_branch.get_by_str_key("meas").unwrap() else {
            panic!("expected a leaf at register 'meas'");
        };
        assert!(matches!(meas_tt.dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(meas_tt.shape, vec![Dim::Fixed(1024), Dim::Fixed(5)]);
        assert!(meas_tt.broadcastable);
    }

    #[test]
    fn test_multiple_circuits() {
        // Two circuits with different register layouts, addressable by index.
        let sl = ShotLoop::new(
            vec![
                circuit_with_cregs(vec![ClassicalRegister::new_owning("c", 2)]),
                circuit_with_cregs(vec![ClassicalRegister::new_owning("d", 4)]),
            ],
            42,
        );

        assert_eq!(sl.input_types().len(), 2);
        assert_eq!(sl.output_types().len(), 2);

        // Inputs: one leaf per circuit.
        for i in 0..2 {
            assert!(matches!(
                sl.input_types().get(i).unwrap(),
                DataTree::Leaf(_)
            ));
        }

        // Outputs: each circuit branch has the right register name.
        let DataTree::Leaf(tt0) = sl
            .output_types()
            .get(0)
            .unwrap()
            .get_by_str_key("c")
            .unwrap()
        else {
            panic!("expected leaf at 0.c");
        };
        assert_eq!(tt0.shape, vec![Dim::Fixed(42), Dim::Fixed(2)]);

        let DataTree::Leaf(tt1) = sl
            .output_types()
            .get(1)
            .unwrap()
            .get_by_str_key("d")
            .unwrap()
        else {
            panic!("expected leaf at 1.d");
        };
        assert_eq!(tt1.shape, vec![Dim::Fixed(42), Dim::Fixed(4)]);
    }

    #[test]
    fn test_accessors() {
        let circuit = circuit_with_cregs(vec![ClassicalRegister::new_owning("c", 1)]);
        let sl = ShotLoop::new(vec![circuit], 7);
        assert_eq!(sl.shots(), 7);
        assert_eq!(sl.circuits().len(), 1);
        assert_eq!(sl.circuits()[0].cregs().len(), 1);
    }

    // -----------------------------------------------------------------------
    // resolve_types_flat
    // -----------------------------------------------------------------------

    fn f64_params(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape,
            broadcastable: true,
        }
    }

    #[test]
    fn test_resolve_types_flat_unbatched() {
        // shape [n] (no batch prefix) -> output unchanged: [shots, reg_len].
        let circuit = circuit_with_params(1, vec![ClassicalRegister::new_owning("meas", 2)]);
        let sl = ShotLoop::new(vec![circuit], 4000);

        let resolved = sl
            .resolve_types_flat(&[f64_params(vec![Dim::Fixed(1)])])
            .unwrap();
        assert_eq!(resolved.len(), 1);
        assert!(matches!(resolved[0].dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(4000), Dim::Fixed(2)]);
        assert!(resolved[0].broadcastable);
    }

    #[test]
    fn test_resolve_types_flat_single_batch_dim() {
        // shape [10, n] -> output [10, shots, reg_len].
        let circuit = circuit_with_params(1, vec![ClassicalRegister::new_owning("meas", 2)]);
        let sl = ShotLoop::new(vec![circuit], 4000);

        let resolved = sl
            .resolve_types_flat(&[f64_params(vec![Dim::Fixed(10), Dim::Fixed(1)])])
            .unwrap();
        assert_eq!(
            resolved[0].shape,
            vec![Dim::Fixed(10), Dim::Fixed(4000), Dim::Fixed(2)]
        );
    }

    #[test]
    fn test_resolve_types_flat_multi_batch_dim() {
        // shape [10, 20, 30, n] -> output [10, 20, 30, shots, reg_len].
        let circuit = circuit_with_params(1, vec![ClassicalRegister::new_owning("meas", 2)]);
        let sl = ShotLoop::new(vec![circuit], 4000);

        let resolved = sl
            .resolve_types_flat(&[f64_params(vec![
                Dim::Fixed(10),
                Dim::Fixed(20),
                Dim::Fixed(30),
                Dim::Fixed(1),
            ])])
            .unwrap();
        assert_eq!(
            resolved[0].shape,
            vec![
                Dim::Fixed(10),
                Dim::Fixed(20),
                Dim::Fixed(30),
                Dim::Fixed(4000),
                Dim::Fixed(2)
            ]
        );
    }

    #[test]
    fn test_resolve_types_flat_wrong_trailing_dim_errors() {
        // A 1-parameter circuit, but the params tensor's trailing axis is 2.
        let circuit = circuit_with_params(1, vec![]);
        let sl = ShotLoop::new(vec![circuit], 10);

        let err = sl
            .resolve_types_flat(&[f64_params(vec![Dim::Fixed(2)])])
            .unwrap_err();
        assert!(matches!(
            err,
            ShotLoopError::WrongParamShape {
                circuit: 0,
                expected: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_resolve_types_flat_wrong_trailing_dim_with_batch_prefix_errors() {
        // Same as above, but with a batch prefix present: the trailing axis is still
        // what's validated against num_parameters.
        let circuit = circuit_with_params(1, vec![]);
        let sl = ShotLoop::new(vec![circuit], 10);

        let err = sl
            .resolve_types_flat(&[f64_params(vec![Dim::Fixed(10), Dim::Fixed(2)])])
            .unwrap_err();
        assert!(matches!(
            err,
            ShotLoopError::WrongParamShape {
                circuit: 0,
                expected: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_resolve_types_flat_wrong_dtype_errors() {
        let circuit = circuit_with_params(1, vec![]);
        let sl = ShotLoop::new(vec![circuit], 10);

        let bit_params = TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape: vec![Dim::Fixed(1)],
            broadcastable: true,
        };
        let err = sl.resolve_types_flat(&[bit_params]).unwrap_err();
        assert!(matches!(
            err,
            ShotLoopError::WrongParamDType { circuit: 0, .. }
        ));
    }

    #[test]
    fn test_resolve_types_flat_empty_shape_errors() {
        let circuit = circuit_with_params(1, vec![]);
        let sl = ShotLoop::new(vec![circuit], 10);

        let err = sl.resolve_types_flat(&[f64_params(vec![])]).unwrap_err();
        assert!(matches!(err, ShotLoopError::EmptyParamShape { circuit: 0 }));
    }

    #[test]
    fn test_resolve_types_flat_named_trailing_dim_accepted() {
        // A Dim::Named trailing axis is accepted permissively, regardless of the
        // circuit's actual parameter count.
        let circuit = circuit_with_params(3, vec![ClassicalRegister::new_owning("meas", 2)]);
        let sl = ShotLoop::new(vec![circuit], 100);

        let resolved = sl
            .resolve_types_flat(&[f64_params(vec![Dim::Fixed(5), Dim::Named("n".into())])])
            .unwrap();
        assert_eq!(
            resolved[0].shape,
            vec![Dim::Fixed(5), Dim::Fixed(100), Dim::Fixed(2)]
        );
    }

    #[test]
    fn test_resolve_types_flat_multiple_circuits_independent_batches() {
        // Two circuits with different batch shapes resolve independently.
        let circuit0 = circuit_with_params(1, vec![ClassicalRegister::new_owning("c", 2)]);
        let circuit1 = circuit_with_params(2, vec![ClassicalRegister::new_owning("d", 4)]);
        let sl = ShotLoop::new(vec![circuit0, circuit1], 50);

        let resolved = sl
            .resolve_types_flat(&[
                f64_params(vec![Dim::Fixed(1)]),
                f64_params(vec![Dim::Fixed(7), Dim::Fixed(2)]),
            ])
            .unwrap();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].shape, vec![Dim::Fixed(50), Dim::Fixed(2)]);
        assert_eq!(
            resolved[1].shape,
            vec![Dim::Fixed(7), Dim::Fixed(50), Dim::Fixed(4)]
        );
    }
}
