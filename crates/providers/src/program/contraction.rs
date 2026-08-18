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

//! Contract a quantum progrom into specified regions.

use hashbrown::HashMap;
use thiserror::Error;

use super::program_function::{NodeId, NodeRole, NodeView, ProgramFunction, Value};
use super::quantum_program::{FunctionId, QuantumProgram};

/// Why a program could not be contracted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ContractionError {
    /// The entry point already calls another function.
    #[error("the entry point calls another function at node {node}")]
    EntryPointCall { node: NodeId },

    /// Two execution resources declare the same node type.
    #[error("resource {first} and resource {second} both handle {name}")]
    RepeatedNodeType {
        name: String,
        first: usize,
        second: usize,
    },
}

/// Rewrite `program` so that its nodes are grouped into functions of one category each, and report
/// the execution resource each function belongs to.
///
/// `resources` declares, per execution resource, the node type names that resource handles, as
/// [`OpNodeType::full_name`](crate::OpNodeType::full_name) renders them. Node types no resource declares
/// share one fallback category. Boundary nodes are matched by role, not by name: a parameter stays in
/// the entry point that declares it. One input may fan out to several units.
///
/// The rewritten program holds one function per unit, ordered so that no unit depends on a later
/// one, and then an entry point that declares the same inputs and outputs as `program`'s and holds
/// one call per unit. Each unit can be run as a whole, because no path leaving a unit leads back
/// into it. Every other function `program` holds is dropped; a contractible entry point calls none
/// of them.
///
/// The table gives each function's resource, indexed by [`FunctionId::index`], and is `None` for a
/// fallback unit and for the entry point. It is not grouped by resource. A dependency may run from
/// one unit of a resource to another, so one resource's functions cannot be run as a batch.
///
/// This is a helper for a `BackendV3` implementation, not part of the interface one must satisfy.
///
/// # Errors
///
/// Returns an error if `program`'s entry point already contains a call node, or if two resources
/// declare the same node type.
pub fn contract<R, N>(
    program: &QuantumProgram,
    resources: R,
) -> Result<(QuantumProgram, Vec<Option<usize>>), ContractionError>
where
    R: IntoIterator,
    R::Item: IntoIterator<Item = N>,
    N: AsRef<str>,
{
    let categories = categories(resources)?;
    let entry = program.entry_function();
    if let Some(node) = entry
        .iter_nodes()
        .find(|node| node.role() == NodeRole::Call)
    {
        // We disallow contracting programs that already have multiple functions to simplify the
        // implementation. this slightly more complicated case could be implemented, if needed.
        return Err(ContractionError::EntryPointCall { node: node.id() });
    }

    let units = partition(entry, &categories);
    let mut table: Vec<Option<usize>> = units.iter().map(|unit| unit.category).collect();
    table.push(None);

    let contracted = QuantumProgram::new(
        assemble(entry, &units),
        program.input_structure().clone(),
        program.output_structure().clone(),
    )
    .expect("the entry point keeps its slots, and each call is built from its callee's signature");
    Ok((contracted, table))
}

/// One unit of work: the nodes that become one function of the contracted program.
struct Unit {
    /// The resource this unit's node types belong to, or `None` for the fallback category.
    category: Option<usize>,
    /// The nodes in topological order of dataflow.
    nodes: Vec<NodeId>,
}

/// Which resource handles each declared node type, keyed by full name.
type Categories = HashMap<String, usize>;

/// Where each value of the function being contracted landed in the function being built.
type Moved = HashMap<Value, Value>;

/// Index the declared node type names by resource.
///
/// A name matches a node type by exact equality with its
/// [`full_name`](crate::OpNodeType::full_name), so a caller depends on no type this crate defines.
fn categories<R, N>(resources: R) -> Result<Categories, ContractionError>
where
    R: IntoIterator,
    R::Item: IntoIterator<Item = N>,
    N: AsRef<str>,
{
    let mut categories = Categories::new();
    for (resource, names) in resources.into_iter().enumerate() {
        for name in names {
            let previous = categories.insert(name.as_ref().to_string(), resource);
            if let Some(first) = previous
                && first != resource
            {
                return Err(ContractionError::RepeatedNodeType {
                    name: name.as_ref().to_string(),
                    first,
                    second: resource,
                });
            }
        }
    }
    Ok(categories)
}

/// Group the computing nodes of `entry` into units.
///
/// This function is a greedy heuristic that aims to result in a few large units rather than
/// the fewest possible.
///
/// A node is placed one layer past any operand of another category, and in the same layer as
/// operands of its own, so every dependency between two units crosses layers. Grouping by layer and
/// category therefore gives units that run in layer order and that no path can re-enter. Every node
/// of one category sharing a layer ends up in one unit, whether or not they depend on each other.
fn partition(entry: &ProgramFunction, categories: &Categories) -> Vec<Unit> {
    // The layer and category of each node, for the nodes that have them.
    let mut placed: Vec<Option<(usize, Option<usize>)>> = vec![None; entry.node_count()];
    let mut groups: Vec<(usize, Unit)> = Vec::new();
    let mut index: HashMap<(usize, Option<usize>), usize> = HashMap::new();

    for node in entry.iter_nodes() {
        let NodeView::Op(op_node_type) = node.view() else {
            continue;
        };
        let category = categories.get(&op_node_type.full_name()).copied();
        let layer = node
            .operands()
            .iter()
            .filter_map(|value| placed[value.node().index()])
            .map(|(layer, of)| if of == category { layer } else { layer + 1 })
            .max()
            .unwrap_or(0);
        placed[node.id().index()] = Some((layer, category));

        let unit = *index.entry((layer, category)).or_insert_with(|| {
            groups.push((
                layer,
                Unit {
                    category,
                    nodes: Vec::new(),
                },
            ));
            groups.len() - 1
        });
        groups[unit].1.nodes.push(node.id());
    }

    // Units were created in the order their first node appears, and this sort is stable, so units
    // sharing a layer keep that order.
    groups.sort_by_key(|&(layer, _)| layer);
    groups.into_iter().map(|(_, unit)| unit).collect()
}

/// Which unit holds each node of `entry`, indexed by node id. A boundary node belongs to none.
fn unit_of(entry: &ProgramFunction, units: &[Unit]) -> Vec<Option<usize>> {
    let mut held = vec![None; entry.node_count()];
    for (index, unit) in units.iter().enumerate() {
        for &node in &unit.nodes {
            held[node.index()] = Some(index);
        }
    }
    held
}

/// Per unit, the values it reads from outside itself and the values it produces that something
/// outside it reads.
///
/// Both are in the order `entry` produced them, since a value sorts by its producer and node ids are
/// issued in that order. Neither repeats a value: a value two nodes of one unit read is one
/// parameter, and a value two other units read is one result.
fn boundary_values(
    entry: &ProgramFunction,
    held: &[Option<usize>],
    units: usize,
) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let mut inputs = vec![Vec::new(); units];
    let mut outputs = vec![Vec::new(); units];

    for node in entry.iter_nodes() {
        let consumer = held[node.id().index()];
        for &value in node.operands() {
            let producer = held[value.node().index()];
            if producer == consumer {
                continue;
            }
            if let Some(consumer) = consumer {
                inputs[consumer].push(value);
            }
            if let Some(producer) = producer {
                outputs[producer].push(value);
            }
        }
    }

    for values in inputs.iter_mut().chain(&mut outputs) {
        values.sort_unstable();
        values.dedup();
    }
    (inputs, outputs)
}

/// Build one function per unit, and then the entry point that calls each of them once.
fn assemble(entry: &ProgramFunction, units: &[Unit]) -> Vec<ProgramFunction> {
    let held = unit_of(entry, units);
    let (inputs, outputs) = boundary_values(entry, &held, units.len());
    let type_of = |value| {
        entry
            .type_of(value)
            .expect("a value of the function being contracted always exists")
            .clone()
    };

    // Each unit takes a parameter per value it reads from outside itself.
    let mut functions: Vec<ProgramFunction> = Vec::with_capacity(units.len() + 1);
    let mut moved: Vec<Moved> = Vec::with_capacity(units.len());
    for unit in &inputs {
        let mut function = ProgramFunction::new();
        let values = unit
            .iter()
            .map(|&value| (value, function.add_parameter(type_of(value))))
            .collect();
        functions.push(function);
        moved.push(values);
    }

    // Each node is rebuilt in its unit from the node type it applies, over the values its operands
    // landed on. An operand is produced within the unit or before it, because no path re-enters a
    // unit.
    for node in entry.iter_nodes() {
        let NodeView::Op(op_node_type) = node.view() else {
            continue;
        };
        let unit = held[node.id().index()].expect("every computing node belongs to a unit");
        let operands: Vec<Value> = node
            .operands()
            .iter()
            .map(|value| moved[unit][value])
            .collect();
        let produced = functions[unit]
            .add_boxed_node(op_node_type.to_owned(), &operands)
            .expect("inference is monomorphic, so a carried node types as it typed before");
        moved[unit].extend(node.outputs().zip(produced));
    }

    for (unit, function) in functions.iter_mut().enumerate() {
        for value in &outputs[unit] {
            function
                .add_result(moved[unit][value])
                .expect("a unit returns a value produced within it");
        }
    }

    // The entry point declares what the function being contracted declared, and computes it by
    // calling each unit in turn.
    let mut new_entry = ProgramFunction::new();
    let mut lifted: Moved = entry
        .parameter_values()
        .map(|value| (value, new_entry.add_parameter(type_of(value))))
        .collect();
    for (unit, function) in functions.iter().enumerate() {
        let operands: Vec<Value> = inputs[unit].iter().map(|value| lifted[value]).collect();
        let produced = new_entry
            .add_call(
                FunctionId::from_index(unit),
                &function.signature(),
                &operands,
            )
            .expect("a call built from its callee's own signature type-checks");
        lifted.extend(outputs[unit].iter().copied().zip(produced));
    }
    for value in entry.result_values() {
        new_entry
            .add_result(lifted[&value])
            .expect("a result of the function being contracted is a parameter or a unit's result");
    }

    functions.push(new_entry);
    functions
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data_tree::DataTree;
    use crate::nodes::{Add, Constant, Mean, Multiply, OpNodeType};
    use crate::tensor::{DType, Dim, Tensor, TensorType};

    /// The type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    /// `functions` as a program whose slots are unnamed: one sequence of leaves per side.
    fn program(functions: Vec<ProgramFunction>) -> QuantumProgram {
        let entry = functions.last().expect("a program holds a function");
        let positional = |count| DataTree::sequence(std::iter::repeat_n(DataTree::Leaf(()), count));
        let inputs = positional(entry.parameters().len());
        let outputs = positional(entry.results().len());
        QuantumProgram::new(functions, inputs, outputs).unwrap()
    }

    /// Evaluate `program` on `args`, one per parameter, returning one tensor per result.
    fn eval(program: &QuantumProgram, args: impl IntoIterator<Item = Tensor>) -> Vec<Tensor> {
        let inputs = DataTree::sequence(args.into_iter().map(DataTree::Leaf));
        program.eval(inputs).unwrap().into_leaves().collect()
    }

    /// The type name of every node of `function`, in order.
    fn names(function: &ProgramFunction) -> Vec<String> {
        function.iter_nodes().map(|node| node.full_name()).collect()
    }

    /// `f(x, y) = x + y` over `len`-element `F64` tensors.
    fn add_function(len: usize) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(len));
        let y = function.add_parameter(f64_1d(len));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();
        function
    }

    // ---------------------------------------------------------------------------
    // Grouping
    // ---------------------------------------------------------------------------

    #[test]
    fn a_program_of_one_category_becomes_one_call() {
        let original = program(vec![add_function(1)]);
        // A boundary node reports a type name like any other node, and declaring those names changes
        // nothing: a boundary node is matched by its role.
        let resources = [vec!["qiskit.add", "qiskit.parameter", "qiskit.result"]];
        let (contracted, table) = contract(&original, resources).unwrap();

        assert_eq!(table, [Some(0), None], "one unit, then the entry point");
        assert_eq!(
            table.len(),
            contracted.functions().len(),
            "the table describes every function of the rewritten program"
        );
        assert_eq!(
            names(contracted.entry_function()),
            [
                "qiskit.parameter",
                "qiskit.parameter",
                "qiskit.call",
                "qiskit.result"
            ],
            "the entry point holds its own boundary and one call, and no arithmetic"
        );
        assert_eq!(
            names(contracted.function(FunctionId::from_index(0)).unwrap()),
            [
                "qiskit.parameter",
                "qiskit.parameter",
                "qiskit.add",
                "qiskit.result"
            ],
            "a boundary node is left where it is, which matching on type names would not do"
        );

        let arguments = [Tensor::from([1.5_f64]), Tensor::from([2.5_f64])];
        assert_eq!(
            eval(&contracted, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn independent_nodes_of_one_category_become_one_unit() {
        // Two sums over four parameters, with no path between them.
        let mut function = ProgramFunction::new();
        let values: Vec<Value> = (0..4).map(|_| function.add_parameter(f64_1d(1))).collect();
        let first = function.add_node(Add, &[values[0], values[1]]).unwrap()[0];
        let second = function.add_node(Add, &[values[2], values[3]]).unwrap()[0];
        function.add_result(first).unwrap();
        function.add_result(second).unwrap();
        let original = program(vec![function]);

        let (contracted, table) = contract(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(
            table,
            [Some(0), None],
            "independent work of one category is one unit, and so one submission"
        );
        let unit = contracted.function(FunctionId::from_index(0)).unwrap();
        assert_eq!(unit.parameters().len(), 4);
        assert_eq!(unit.results().len(), 2);

        let arguments = [1.0, 2.0, 30.0, 40.0].map(|x| Tensor::from([x]));
        assert_eq!(
            eval(&contracted, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn a_dependency_of_another_category_splits_a_category_in_two() {
        // `add` twice over a `mean` between them, so the two sums cannot share a unit. A unit the
        // mean's result re-entered could not be run whole.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        let mean = function.add_node(Mean::new(0), &[sum]).unwrap()[0];
        let doubled = function.add_node(Add, &[mean, mean]).unwrap()[0];
        function.add_result(doubled).unwrap();
        let original = program(vec![function]);

        let (contracted, table) = contract(&original, [["qiskit.add"], ["qiskit.mean"]]).unwrap();

        assert_eq!(
            table,
            [Some(0), Some(1), Some(0), None],
            "the first resource runs twice, either side of the second"
        );
        let arguments = [Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f64, 20.0])];
        assert_eq!(
            eval(&contracted, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn node_types_no_resource_handles_share_one_category() {
        // `multiply` and `mean` are undeclared and independent, so they share the fallback category
        // and therefore one unit.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        let product = function.add_node(Multiply, &[x, y]).unwrap()[0];
        let mean = function.add_node(Mean::new(0), &[y]).unwrap()[0];
        function.add_result(sum).unwrap();
        function.add_result(product).unwrap();
        function.add_result(mean).unwrap();
        let original = program(vec![function]);

        let (contracted, table) = contract(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [Some(0), None, None]);
        assert_eq!(
            names(contracted.function(FunctionId::from_index(1)).unwrap()),
            [
                "qiskit.parameter",
                "qiskit.parameter",
                "qiskit.multiply",
                "qiskit.mean",
                "qiskit.result",
                "qiskit.result"
            ],
            "one unit holds both undeclared node types"
        );

        let arguments = [Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f64, 20.0])];
        assert_eq!(
            eval(&contracted, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn one_input_may_fan_out_to_several_units() {
        // Both units read `x`, and the sum reads it twice.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, x]).unwrap()[0];
        let product = function.add_node(Multiply, &[x, x]).unwrap()[0];
        function.add_result(sum).unwrap();
        function.add_result(product).unwrap();
        let original = program(vec![function]);

        let (contracted, table) = contract(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [Some(0), None, None]);
        for unit in 0..2 {
            assert_eq!(
                contracted
                    .function(FunctionId::from_index(unit))
                    .unwrap()
                    .parameters()
                    .len(),
                1,
                "a value a unit reads twice is one parameter"
            );
        }
        assert_eq!(
            contracted
                .entry_function()
                .iter_nodes()
                .filter(|node| node.role() == NodeRole::Call)
                .count(),
            2,
            "the entry point passes its one parameter to both calls"
        );

        let arguments = [Tensor::from([3.0_f64, 4.0])];
        assert_eq!(
            eval(&contracted, arguments.clone()),
            eval(&original, arguments)
        );
    }

    // ---------------------------------------------------------------------------
    // What the partition guarantees
    // ---------------------------------------------------------------------------

    /// Assert that no node reads a unit later than its own, and that a unit holds one category.
    ///
    /// This is the whole of what makes each unit runnable in one go, in order. Leaving a unit moves
    /// strictly forward, so no path can lead back into one, and running the units in order never
    /// needs a value a later unit produces.
    fn assert_runnable_in_order(entry: &ProgramFunction, units: &[Unit], categories: &Categories) {
        let held = unit_of(entry, units);
        for node in entry.iter_nodes() {
            let Some(consumer) = held[node.id().index()] else {
                continue;
            };
            for value in node.operands() {
                if let Some(producer) = held[value.node().index()] {
                    assert!(
                        producer <= consumer,
                        "node {} of unit {consumer} reads unit {producer}",
                        node.id()
                    );
                }
            }
        }
        assert_eq!(
            units.iter().map(|unit| unit.nodes.len()).sum::<usize>(),
            entry
                .iter_nodes()
                .filter(|node| node.role() == NodeRole::Op)
                .count(),
            "every computing node is held by exactly one unit"
        );
        for (index, unit) in units.iter().enumerate() {
            assert!(!unit.nodes.is_empty(), "unit {index} holds nothing");
            for &id in &unit.nodes {
                let NodeView::Op(op_node_type) = entry.node(id).unwrap().view() else {
                    panic!("unit {index} holds the boundary node {id}")
                };
                assert_eq!(
                    categories.get(&op_node_type.full_name()).copied(),
                    unit.category,
                    "node {id} of unit {index} is of another category"
                );
            }
        }
    }

    #[test]
    fn every_dependency_between_two_units_runs_forwards() {
        // Two categories interleaved over two diamonds, so a unit is entered from more than one
        // unit and left towards more than one.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        let product = function.add_node(Multiply, &[x, y]).unwrap()[0];
        let both = function.add_node(Add, &[sum, product]).unwrap()[0];
        let scaled = function.add_node(Multiply, &[sum, sum]).unwrap()[0];
        let total = function.add_node(Add, &[both, scaled]).unwrap()[0];
        function.add_result(total).unwrap();
        function.add_result(scaled).unwrap();

        let categories = categories([["qiskit.add"], ["qiskit.multiply"]]).unwrap();
        let units = partition(&function, &categories);

        assert_eq!(
            units.iter().map(|unit| unit.category).collect::<Vec<_>>(),
            [Some(0), Some(1), Some(0), Some(1), Some(0)],
            "each category runs three times and twice, alternating"
        );
        assert_runnable_in_order(&function, &units, &categories);

        let (inputs, outputs) =
            boundary_values(&function, &unit_of(&function, &units), units.len());
        assert_eq!(
            outputs[0].len(),
            1,
            "a value two later units read is returned once"
        );
        assert_eq!(inputs[0], [x, y]);
    }

    #[test]
    fn a_unit_is_left_where_a_category_reappears_later() {
        // A chain of alternating categories, which is the worst case for the heuristic: every node
        // is its own unit.
        let mut function = ProgramFunction::new();
        let mut value = function.add_parameter(f64_1d(2));
        for step in 0..6 {
            value = if step % 2 == 0 {
                function.add_node(Add, &[value, value]).unwrap()[0]
            } else {
                function.add_node(Multiply, &[value, value]).unwrap()[0]
            };
        }
        function.add_result(value).unwrap();

        let categories = categories([["qiskit.add"], ["qiskit.multiply"]]).unwrap();
        let units = partition(&function, &categories);

        assert_eq!(units.len(), 6, "no two steps of the chain can share a unit");
        assert_runnable_in_order(&function, &units, &categories);
    }

    // ---------------------------------------------------------------------------
    // What the rewritten program keeps
    // ---------------------------------------------------------------------------

    #[test]
    fn the_rewritten_program_computes_and_declares_the_same_thing() {
        // A constant of its own category, so its unit reads nothing at all; a node whose result
        // nothing reads; a parameter returned directly; and one value returned twice.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let two = function
            .add_node(Constant::new(Tensor::from([2.0_f64, 2.0])), &[])
            .unwrap()[0];
        let scaled = function.add_node(Multiply, &[x, two]).unwrap()[0];
        function.add_node(Mean::new(0), &[scaled]).unwrap();
        function.add_result(scaled).unwrap();
        function.add_result(x).unwrap();
        function.add_result(scaled).unwrap();

        let names = DataTree::mapping([
            ("scaled", DataTree::Leaf(())),
            ("original", DataTree::Leaf(())),
            ("again", DataTree::Leaf(())),
        ])
        .unwrap();
        let original =
            QuantumProgram::new(vec![function], DataTree::Leaf(()), names.clone()).unwrap();

        let resources = [vec!["qiskit.multiply"], vec!["qiskit.constant"]];
        let (contracted, table) = contract(&original, resources).unwrap();

        assert_eq!(
            table,
            [Some(1), Some(0), None, None],
            "the constant, the product, the undeclared mean, and the entry point"
        );
        let constant = contracted.function(FunctionId::from_index(0)).unwrap();
        assert!(
            constant.parameters().is_empty(),
            "a unit that reads nothing is called with no operands"
        );
        assert_eq!(
            contracted.entry_function().signature(),
            original.entry_function().signature(),
            "the entry point declares what it declared, in the order it declared it"
        );
        assert_eq!(contracted.input_structure(), &DataTree::Leaf(()));
        assert_eq!(contracted.output_structure(), &names);

        let input = || DataTree::Leaf(Tensor::from([3.0_f64, 4.0]));
        assert_eq!(
            contracted.eval(input()).unwrap(),
            original.eval(input()).unwrap()
        );
    }

    #[test]
    fn a_program_with_nothing_to_contract_keeps_only_its_entry_point() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        function.add_result(x).unwrap();
        let original = program(vec![function]);

        let (contracted, table) = contract(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [None]);
        assert_eq!(contracted.functions().len(), 1);
        assert_eq!(
            eval(&contracted, [Tensor::from([1.0_f64])]),
            [Tensor::from([1.0_f64])]
        );
    }

    #[test]
    fn a_function_the_entry_point_cannot_reach_is_dropped() {
        let original = program(vec![add_function(4), add_function(1)]);
        assert_eq!(original.functions().len(), 2);

        let (contracted, table) = contract(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [Some(0), None], "the unit, then the entry point");
        assert_eq!(
            contracted
                .function(FunctionId::from_index(0))
                .unwrap()
                .signature(),
            add_function(1).signature(),
            "@0 is the unit, not the function that was @0"
        );
    }

    /// A node type with two results: the sum of its operands, and their difference.
    #[derive(Clone)]
    struct SumAndDifference;

    impl OpNodeType for SumAndDifference {
        type Error = std::convert::Infallible;

        fn name(&self) -> &str {
            "sum_and_difference"
        }
        fn namespace(&self) -> &str {
            "vendor"
        }
        fn arity(&self) -> usize {
            2
        }
        fn has_builtin_eval(&self) -> bool {
            true
        }
        fn infer_output_types(
            &self,
            inputs: &[TensorType],
        ) -> Result<Vec<TensorType>, Self::Error> {
            Ok(vec![inputs[0].clone(), inputs[0].clone()])
        }
        fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
            let [x, y] = args else { panic!("two operands") };
            Ok(vec![x.add_tensor(y).unwrap(), x.sub_tensor(y).unwrap()])
        }
    }

    #[test]
    fn a_result_other_than_the_first_crosses_a_unit_boundary_as_itself() {
        // Only the difference is read by the other unit, so the value that crosses is the second
        // result of its producer, and the unit returns that one alone.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let both = function.add_node(SumAndDifference, &[x, y]).unwrap();
        let scaled = function.add_node(Multiply, &[both[1], both[1]]).unwrap()[0];
        function.add_result(both[0]).unwrap();
        function.add_result(scaled).unwrap();
        let original = program(vec![function]);

        let resources = [vec!["vendor.sum_and_difference"], vec!["qiskit.multiply"]];
        let (contracted, table) = contract(&original, resources).unwrap();

        assert_eq!(table, [Some(0), Some(1), None]);
        let first = contracted.function(FunctionId::from_index(0)).unwrap();
        assert_eq!(
            first.results().len(),
            2,
            "the sum is returned for the entry point and the difference for the second unit"
        );
        assert_eq!(
            contracted
                .function(FunctionId::from_index(1))
                .unwrap()
                .signature()
                .inputs
                .len(),
            1,
            "the second unit reads one value, whatever slot it came from"
        );

        let arguments = [Tensor::from([10.0_f64, 20.0]), Tensor::from([1.0_f64, 2.0])];
        assert_eq!(
            eval(&contracted, arguments.clone()),
            eval(&original, arguments)
        );
    }

    // ---------------------------------------------------------------------------
    // Rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn an_entry_point_that_already_calls_a_function_is_rejected() {
        // Contracting twice would put calls into units, so the second attempt is refused rather
        // than returning a program whose units call units.
        let callee = add_function(1);
        let signature = callee.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let y = entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &signature, &[x, y])
            .unwrap()[0];
        entry.add_result(called).unwrap();
        let original = program(vec![callee, entry]);

        let Err(err) = contract(&original, [["qiskit.add"]]) else {
            panic!("a program holding a call is already contracted")
        };
        let call = original
            .entry_function()
            .iter_nodes()
            .find(|node| node.role() == NodeRole::Call)
            .expect("the entry point holds a call")
            .id();
        assert_eq!(err, ContractionError::EntryPointCall { node: call });
        assert_eq!(
            err.to_string(),
            "the entry point calls another function at node 2"
        );
    }

    #[test]
    fn two_resources_cannot_declare_one_node_type() {
        let original = program(vec![add_function(1)]);

        let resources = [vec!["qiskit.add"], vec!["qiskit.mean", "qiskit.add"]];
        let Err(err) = contract(&original, resources) else {
            panic!("a node type belongs to one resource")
        };
        assert_eq!(
            err,
            ContractionError::RepeatedNodeType {
                name: "qiskit.add".to_string(),
                first: 0,
                second: 1,
            }
        );
        assert_eq!(
            err.to_string(),
            "resource 0 and resource 1 both handle qiskit.add"
        );

        assert!(
            contract(&original, [["qiskit.add", "qiskit.add"]]).is_ok(),
            "one resource may name a node type twice"
        );
    }
}
