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

//! Contracting a [`QuantumProgram`] into single-category regions.
//!
//! See [`QuantumProgram::into_regions`].

use super::*;
use hashbrown::HashSet;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::visit::{EdgeRef, NodeIndexable};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One region produced by [`QuantumProgram::into_regions`].
#[derive(Clone, Debug)]
pub struct ProgramRegion {
    /// The region node's label in [`ContractedProgram::program`].
    pub label: String,
    /// Index into the `categories` argument of [`QuantumProgram::into_regions`], or `None`
    /// if this region holds nodes whose type appeared in no category.
    pub category: Option<usize>,
    /// The labels this region's nodes had in the original program, in topological order.
    pub members: Vec<String>,
    /// The region program's input keys paired with the member input port each one feeds,
    /// in the leaf order of the region node's `input_types()`.
    pub input_ports: Vec<(String, Port)>,
    /// The region program's output keys paired with the member output port each one comes
    /// from, in the leaf order of the region node's `output_types()`.
    pub output_ports: Vec<(String, Port)>,
}

/// The result of [`QuantumProgram::into_regions`].
pub struct ContractedProgram {
    /// A program equivalent to the one that was consumed, in which every node other than
    /// the hoisted [`Input`] sources is itself a [`QuantumProgram`]. Reach inside one with
    /// `program.get_node(label).and_then(ProgramNode::as_quantum_program)`.
    pub program: QuantumProgram,
    /// One entry per region, ordered so that region `i` never depends on region `j > i`
    /// (i.e. a valid topological order of the contracted graph).
    pub regions: Vec<ProgramRegion>,
}

/// Errors returned by [`QuantumProgram::into_regions`].
#[derive(Debug, Error)]
pub enum ContractionError {
    /// The program's graph contains a cycle, so it has no topological order.
    #[error("quantum program graph contains a cycle")]
    Cycle,

    /// Rebuilding the contracted program failed. This indicates a bug in the contraction
    /// itself; it is not reachable for a program that could be built in the first place.
    #[error(transparent)]
    Build(#[from] QuantumProgramError),
}

// ---------------------------------------------------------------------------
// into_regions
// ---------------------------------------------------------------------------

impl QuantumProgram {
    /// Partition this program into regions of same-category nodes, returning an equivalent
    /// program whose nodes are those regions.
    ///
    /// Each entry of `categories` is a set of node type names — [`ProgramNode::full_name`]
    /// values such as `"qiskit.shot_loop"`, matched by exact string equality. A typical
    /// caller is a backend's `run` method, where each entry lists the node types one runner
    /// supports; contracting the submitted program then yields one region per submission,
    /// with [`ProgramRegion::category`] naming the runner to send it to. Node types that
    /// appear in no entry share a single fallback category, reported as
    /// [`ProgramRegion::category`] `== None`.
    ///
    /// `self` is consumed because member nodes are *moved* into the region programs.
    ///
    /// # Region shapes
    ///
    /// Regions are always *convex*: a region can never be re-entered by a path that leaves
    /// it, so the contracted graph is guaranteed to be acyclic and every region can be
    /// executed as a unit. Within that constraint the partition is a linear-time heuristic
    /// that aims for few, large regions but is not guaranteed to be optimal — for example,
    /// a node is never pulled *backwards* into an earlier region it could have joined.
    ///
    /// Two guarantees worth calling out, both intentional:
    ///
    /// - Nodes with **no path between them** may still be merged, so two independent nodes
    ///   of the same category typically end up in one region (and hence one submission).
    /// - A node type listed in several categories is genuinely flexible: it is placed
    ///   wherever it reduces the region count, falling back to the earliest listing only to
    ///   break ties. The fallback category is never admissible for a listed type, since
    ///   "no runner supports this" is a different statement from "some runner does".
    ///
    /// # Program inputs
    ///
    /// The [`Input`] source nodes created by [`Self::add_input`] are *not* region members;
    /// they are hoisted into the returned program under their original labels so that a
    /// single program input can still fan out to several regions.
    ///
    /// # Region I/O keys
    ///
    /// A region's inputs and outputs are keyed as `"{member_label}.{port_path}"` (or just
    /// `"{member_label}"` for a root port), with `#1`, `#2`, … appended to break the
    /// occasional collision. These keys are *not* safe to feed to
    /// [`DataTree::get_by_str_key`](crate::DataTree::get_by_str_key), which splits on `.`;
    /// use [`ProgramRegion::input_ports`]/[`ProgramRegion::output_ports`] instead of
    /// parsing them.
    ///
    /// # Ways the result differs from the original
    ///
    /// The contracted program's `input_types()` and `output_types()` are identical to the
    /// original's, key order included, and it computes the same values. But:
    ///
    /// - [`ProgramNode::implements_call`] on a region node is always `true`, even for a
    ///   region full of nodes that cannot execute locally. Dispatch on
    ///   [`ProgramRegion::category`], not on `implements_call()`.
    /// - A region with no outputs at all is skipped wholesale at call time, whereas the
    ///   original would have called its members. Since nodes are pure, the only observable
    ///   difference is that an error raised by such a dead node disappears.
    /// - Wiring is not validated, so an incomplete program contracts successfully; the
    ///   resulting error is merely nested one level deeper (a
    ///   [`QuantumProgramCallError::NodeCall`] naming the region, wrapping the original
    ///   [`QuantumProgramCallError::UnwiredInput`]).
    pub fn into_regions(
        self,
        categories: &[Vec<String>],
    ) -> Result<ContractedProgram, ContractionError> {
        // -------------------------------------------------------------------
        // Phase A: plan everything while `self` is still intact.
        // -------------------------------------------------------------------
        let n_bound = self.graph.node_bound();

        // Invert `label_to_node`. Note that `label_to_node` is never iterated for anything
        // order-dependent, since it is a hash map.
        let mut labels: Vec<String> = vec![String::new(); n_bound];
        for (label, &idx) in &self.label_to_node {
            labels[idx.index()] = label.clone();
        }

        // The nodes backing `add_input`-declared program inputs are hoisted into the outer
        // program rather than joining a region. They have no input leaves, so no path can
        // pass through one and they simply become sources of the contracted graph.
        let mut hoisted = vec![false; n_bound];
        for (_, idx) in &self.input_source_nodes {
            hoisted[idx.index()] = true;
        }

        // Map each node type to every category listing it, in order. `fallback` is the
        // synthetic colour shared by all types that appear in none.
        let fallback = categories.len();
        let mut listed_in: HashMap<&str, Vec<usize>> = HashMap::new();
        for (category, names) in categories.iter().enumerate() {
            for name in names {
                let entry = listed_in.entry(name.as_str()).or_default();
                if entry.last() != Some(&category) {
                    entry.push(category);
                }
            }
        }
        let admissible: Vec<Vec<usize>> = (0..n_bound)
            .map(|i| {
                let full_name = self.graph[NodeIndex::new(i)].node.full_name();
                listed_in
                    .get(full_name.as_str())
                    .cloned()
                    .unwrap_or_else(|| vec![fallback])
            })
            .collect();

        let topo_order: Vec<NodeIndex> = lexicographical_topological_sort(
            &self.graph,
            |n: NodeIndex| Ok::<usize, std::convert::Infallible>(n.index()),
            false,
            None,
        )
        .map_err(|_| ContractionError::Cycle)?;
        // See `topo_plan`: a truncated order is how a cycle surfaces.
        if topo_order.len() != self.graph.node_count() {
            return Err(ContractionError::Cycle);
        }

        let assignment = assign_regions(
            &self.graph,
            &topo_order,
            &admissible,
            &hoisted,
            fallback + 1,
        );
        let region_of = &assignment.region_of;
        let n_regions = assignment.members.len();
        let crosses =
            |src: NodeIndex, tgt: NodeIndex| region_of[src.index()] != region_of[tgt.index()];

        // Snapshot the edges. `edge_references` yields them in `EdgeIndex` (i.e. insertion)
        // order, so replaying it below is deterministic.
        let edges: Vec<EdgeInfo> = self
            .graph
            .edge_references()
            .map(|e| EdgeInfo {
                src: e.source(),
                from_leaf: e.weight().from_leaf,
                tgt: e.target(),
                to_leaf: e.weight().to_leaf,
            })
            .collect();

        // Which member ports sit on a region boundary and so need a region I/O key. An
        // input leaf needs one if it is fed from outside its region; an output leaf needs
        // one if anything outside its region reads it.
        let mut needs_in: HashSet<(NodeIndex, LeafIdx)> = HashSet::new();
        let mut needs_out: HashSet<(NodeIndex, LeafIdx)> = HashSet::new();
        for e in &edges {
            if crosses(e.src, e.tgt) {
                if !hoisted[e.src.index()] {
                    needs_out.insert((e.src, e.from_leaf));
                }
                needs_in.insert((e.tgt, e.to_leaf));
            }
        }
        for (_, idx, leaf) in &self.inputs {
            needs_in.insert((*idx, *leaf));
        }
        for (_, idx, leaf) in &self.outputs {
            if !hoisted[idx.index()] {
                needs_out.insert((*idx, *leaf));
            }
        }

        // Generate region I/O keys by sweeping members in topological order and their
        // leaves in index order, which is both deterministic and a natural I/O ordering.
        let mut in_key: HashMap<(NodeIndex, LeafIdx), String> = HashMap::new();
        let mut out_key: HashMap<(NodeIndex, LeafIdx), String> = HashMap::new();
        let mut regions: Vec<ProgramRegion> = Vec::with_capacity(n_regions);
        for (rid, members) in assignment.members.iter().enumerate() {
            let mut seen_in: HashSet<String> = HashSet::new();
            let mut seen_out: HashSet<String> = HashSet::new();
            let mut input_ports: Vec<(String, Port)> = Vec::new();
            let mut output_ports: Vec<(String, Port)> = Vec::new();
            for &v in members {
                let view = &self.graph[v];
                let label = labels[v.index()].as_str();
                for (leaf, path) in view.input_leaf_paths.iter().enumerate() {
                    if needs_in.contains(&(v, leaf)) {
                        let key = unique_port_key(&mut seen_in, label, path);
                        in_key.insert((v, leaf), key.clone());
                        input_ports.push((key, Port::new(label, path.clone())));
                    }
                }
                for (leaf, path) in view.output_leaf_paths.iter().enumerate() {
                    if needs_out.contains(&(v, leaf)) {
                        let key = unique_port_key(&mut seen_out, label, path);
                        out_key.insert((v, leaf), key.clone());
                        output_ports.push((key, Port::new(label, path.clone())));
                    }
                }
            }
            let colour = assignment.colour_of_region[rid];
            regions.push(ProgramRegion {
                label: format!("region_{rid}"),
                category: (colour != fallback).then_some(colour),
                members: members.iter().map(|&v| labels[v.index()].clone()).collect(),
                input_ports,
                output_ports,
            });
        }

        // Bucket the edges so that each is visited once overall rather than once per region.
        let mut intra_region_edges: Vec<Vec<usize>> = vec![Vec::new(); n_regions];
        let mut crossing_edges: Vec<usize> = Vec::new();
        for (i, e) in edges.iter().enumerate() {
            match region_of[e.src.index()] {
                Some(rid) if !crosses(e.src, e.tgt) => intra_region_edges[rid].push(i),
                _ => crossing_edges.push(i),
            }
        }

        // -------------------------------------------------------------------
        // Phase B: take the nodes out of the old graph.
        // -------------------------------------------------------------------
        let QuantumProgram {
            graph,
            inputs,
            input_source_nodes,
            outputs,
            ..
        } = self;
        // `into_nodes_edges` preserves order, so `views[i]` is `NodeIndex::new(i)`'s view.
        let (raw_nodes, _) = graph.into_nodes_edges();
        let mut views: Vec<Option<NodeView>> =
            raw_nodes.into_iter().map(|n| Some(n.weight)).collect();

        // -------------------------------------------------------------------
        // Phase C: build one program per region.
        // -------------------------------------------------------------------
        let mut region_programs: Vec<QuantumProgram> = Vec::with_capacity(n_regions);
        for (rid, region) in regions.iter().enumerate() {
            let mut inner = QuantumProgram::new();
            for &v in &assignment.members[rid] {
                let view = views[v.index()]
                    .take()
                    .expect("a node belongs to at most one region");
                inner.add_node_view(labels[v.index()].clone(), view)?;
            }
            for &i in &intra_region_edges[rid] {
                let e = &edges[i];
                let from = port_in(
                    &inner,
                    &labels[e.src.index()],
                    e.from_leaf,
                    PortSide::Output,
                );
                let to = port_in(&inner, &labels[e.tgt.index()], e.to_leaf, PortSide::Input);
                inner.add_edge(from, to)?;
            }
            for (key, port) in &region.input_ports {
                inner.set_input(key, port.clone())?;
            }
            for (key, port) in &region.output_ports {
                inner.set_output(key, port.clone())?;
            }
            region_programs.push(inner);
        }

        // -------------------------------------------------------------------
        // Phase D: assemble the outer program. The push order into `inputs`,
        // `input_source_nodes` and `outputs` is preserved throughout, which is what makes
        // the contracted program's `input_types()`/`output_types()` — and hence its flat
        // argument order — identical to the original's.
        // -------------------------------------------------------------------
        let mut outer = QuantumProgram::new();

        for (key, idx) in &input_source_nodes {
            let view = views[idx.index()]
                .take()
                .expect("a hoisted input source node is not a region member");
            let node_idx = outer.add_node_view(labels[idx.index()].clone(), view)?;
            outer.declare_input_source(key.clone(), node_idx);
        }

        // Region programs must be fully built by now: `add_node` snapshots their I/O trees.
        for (region, inner) in regions.iter().zip(region_programs) {
            outer.add_node(region.label.clone(), inner)?;
        }

        // The port on the outer program producing `(node, leaf)`: either a region's output
        // key, or the root output of a hoisted `Input` node.
        let source_port = |node: NodeIndex, leaf: LeafIdx| match region_of[node.index()] {
            Some(rid) => Port::new(
                regions[rid].label.clone(),
                vec![OwnedPathEntry::Key(out_key[&(node, leaf)].clone())],
            ),
            None => Port::new(labels[node.index()].clone(), vec![]),
        };
        // The port on the outer program consuming `(node, leaf)`.
        let target_port = |node: NodeIndex, leaf: LeafIdx| {
            let rid = region_of[node.index()].expect("a consumer port belongs to a region");
            Port::new(
                regions[rid].label.clone(),
                vec![OwnedPathEntry::Key(in_key[&(node, leaf)].clone())],
            )
        };

        for &i in &crossing_edges {
            let e = &edges[i];
            outer.add_edge(
                source_port(e.src, e.from_leaf),
                target_port(e.tgt, e.to_leaf),
            )?;
        }
        for (key, idx, leaf) in &inputs {
            outer.set_input(key, target_port(*idx, *leaf))?;
        }
        for (key, idx, leaf) in &outputs {
            outer.set_output(key, source_port(*idx, *leaf))?;
        }
        // A program declaring only `add_input` inputs never triggers a rebuild above.
        outer.rebuild_io_types();

        Ok(ContractedProgram {
            program: outer,
            regions,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A snapshot of one edge, taken before the graph is consumed.
struct EdgeInfo {
    src: NodeIndex,
    from_leaf: LeafIdx,
    tgt: NodeIndex,
    to_leaf: LeafIdx,
}

/// The port addressing leaf `leaf` on the given side of `prog`'s node labelled `label`.
fn port_in(prog: &QuantumProgram, label: &str, leaf: LeafIdx, side: PortSide) -> Port {
    let idx = prog.label_to_node[label];
    Port::new(label, prog.graph[idx].leaf_paths(side)[leaf].clone())
}

/// Flatten a member port to a region I/O key. Not injective — labels may contain dots —
/// so callers must disambiguate; see [`unique_port_key`].
fn port_key(label: &str, path: &[OwnedPathEntry]) -> String {
    if path.is_empty() {
        label.to_string()
    } else {
        format!("{label}.{}", format_path(path))
    }
}

/// [`port_key`], with a `#1`, `#2`, … suffix appended if needed to keep it unique among
/// `seen`. `seen` must be per region *and* per side: reusing one key on both sides is
/// harmless, since a program's input and output keys live in separate namespaces.
fn unique_port_key(seen: &mut HashSet<String>, label: &str, path: &[OwnedPathEntry]) -> String {
    let base = port_key(label, path);
    if seen.insert(base.clone()) {
        return base;
    }
    let mut n = 1;
    loop {
        let candidate = format!("{base}#{n}");
        if seen.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

// ---------------------------------------------------------------------------
// The partitioning heuristic
// ---------------------------------------------------------------------------

/// The output of [`assign_regions`].
struct RegionAssignment {
    /// Region id per node index; `None` for hoisted nodes.
    region_of: Vec<Option<usize>>,
    /// Member node indices per region id, in topological order.
    members: Vec<Vec<NodeIndex>>,
    /// The colour (category index) of each region.
    colour_of_region: Vec<usize>,
}

/// Partition `graph`'s non-hoisted nodes into convex, single-colour regions.
///
/// Each node is assigned a colour from `admissible[node.index()]` (which must be
/// non-empty, with values below `n_colours`) together with a *depth*, and the regions are
/// the classes of equal `(colour, depth)`. Depth counts colour changes along the longest
/// incoming path:
///
/// ```text
/// d_c(v) = max( M_c, M_≠c + 1 )   where M_c  = max d(u) over preds u coloured c
///                                       M_≠c = max d(u) over preds u coloured otherwise
/// ```
///
/// and the chosen colour is the admissible one minimising `(d_c(v), key_is_new, c)` —
/// shallowest first, then preferring an existing region, then the earliest category.
///
/// Because `d(v) >= d(u) + [c(u) != c(v)]` holds for every edge `u -> v`, `d` is
/// non-decreasing along any path and strictly increases across any region boundary. Hence
/// every region is convex (a path leaving a region cannot return to it) and the contracted
/// graph is acyclic — indeed sorting the region keys by `(depth, colour)`, as done here,
/// is itself a valid topological order of the contracted graph.
///
/// Runs in `O(V + E + sum of |admissible|)`.
fn assign_regions<N, E>(
    graph: &DiGraph<N, E>,
    topo_order: &[NodeIndex],
    admissible: &[Vec<usize>],
    hoisted: &[bool],
    n_colours: usize,
) -> RegionAssignment {
    let n_bound = graph.node_bound();
    let mut colour_of: Vec<usize> = vec![0; n_bound];
    let mut depth_of: Vec<usize> = vec![0; n_bound];

    // `(colour, depth)` keys in discovery order, plus the reverse lookup.
    let mut keys: Vec<(usize, usize)> = Vec::new();
    let mut key_of: HashMap<(usize, usize), usize> = HashMap::new();
    // Per-node key index, so the second pass doesn't have to re-hash.
    let mut node_key: Vec<Option<usize>> = vec![None; n_bound];

    // Scratch reused across nodes: the deepest predecessor of each colour. `touched` lists
    // the entries to reset, so clearing is proportional to the in-degree, not `n_colours`.
    let mut deepest: Vec<Option<usize>> = vec![None; n_colours];
    let mut touched: Vec<usize> = Vec::new();

    for &v in topo_order {
        if hoisted[v.index()] {
            continue;
        }

        touched.clear();
        for u in graph.neighbors_directed(v, Direction::Incoming) {
            if hoisted[u.index()] {
                continue;
            }
            let colour = colour_of[u.index()];
            let depth = depth_of[u.index()];
            match &mut deepest[colour] {
                Some(best) => *best = (*best).max(depth),
                slot => {
                    *slot = Some(depth);
                    touched.push(colour);
                }
            }
        }

        // The two deepest *distinct* colours, which give `M_≠c` in constant time below.
        let mut first: Option<(usize, usize)> = None;
        let mut second: Option<(usize, usize)> = None;
        for &colour in &touched {
            let entry = (
                deepest[colour].expect("touched colours have a depth"),
                colour,
            );
            if first.is_none_or(|best| entry.0 > best.0) {
                second = first;
                first = Some(entry);
            } else if second.is_none_or(|runner_up| entry.0 > runner_up.0) {
                second = Some(entry);
            }
        }

        let mut chosen: Option<(usize, bool, usize)> = None;
        for &colour in &admissible[v.index()] {
            let same = deepest[colour].unwrap_or(0);
            let different = match first {
                Some((depth, c)) if c != colour => Some(depth),
                Some(_) => second.map(|(depth, _)| depth),
                None => None,
            };
            let depth = match different {
                Some(d) => same.max(d + 1),
                None => same,
            };
            let candidate = (depth, !key_of.contains_key(&(colour, depth)), colour);
            if chosen.is_none_or(|best| candidate < best) {
                chosen = Some(candidate);
            }
        }
        let (depth, _, colour) = chosen.expect("every node has at least one admissible colour");

        for &c in &touched {
            deepest[c] = None;
        }

        colour_of[v.index()] = colour;
        depth_of[v.index()] = depth;
        let key = *key_of.entry((colour, depth)).or_insert_with(|| {
            keys.push((colour, depth));
            keys.len() - 1
        });
        node_key[v.index()] = Some(key);
    }

    // Number the regions by increasing `(depth, colour)`, which is a topological order of
    // the contracted graph.
    let mut by_depth: Vec<usize> = (0..keys.len()).collect();
    by_depth.sort_unstable_by_key(|&key| (keys[key].1, keys[key].0));
    let mut region_of_key = vec![0; keys.len()];
    for (rid, &key) in by_depth.iter().enumerate() {
        region_of_key[key] = rid;
    }

    let mut region_of: Vec<Option<usize>> = vec![None; n_bound];
    let mut members: Vec<Vec<NodeIndex>> = vec![Vec::new(); keys.len()];
    for &v in topo_order {
        if let Some(key) = node_key[v.index()] {
            let rid = region_of_key[key];
            region_of[v.index()] = Some(rid);
            members[rid].push(v);
        }
    }

    RegionAssignment {
        colour_of_region: by_depth.iter().map(|&key| keys[key].0).collect(),
        region_of,
        members,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_nodes::{Add, Mean, Multiply};
    use crate::store::Store;
    use crate::tensor::{DType, DTypeLike, Dim};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_store(val: f64) -> Store {
        Store::new(DataTree::new_leaf(Tensor::from([val])))
    }

    fn concrete_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(DType::F64),
            shape: vec![Dim::Fixed(len)],
            broadcastable: true,
        }
    }

    fn cats(lists: &[&[&str]]) -> Vec<Vec<String>> {
        lists
            .iter()
            .map(|names| names.iter().map(|n| n.to_string()).collect())
            .collect()
    }

    /// Compare by `Debug` rendering: neither `Tensor` nor `TensorType` is `PartialEq`.
    fn assert_debug_eq<T: std::fmt::Debug>(actual: &[T], expected: &[T]) {
        let render = |xs: &[T]| xs.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>();
        assert_eq!(render(actual), render(expected));
    }

    /// The top-level key sequence of an I/O tree, i.e. its declaration order.
    fn top_keys<T>(tree: &DataTree<T>) -> Vec<String> {
        tree.iter_children()
            .map(|(key, _)| key.unwrap_or_default().to_string())
            .collect()
    }

    fn region_keys(ports: &[(String, Port)]) -> Vec<&str> {
        ports.iter().map(|(key, _)| key.as_str()).collect()
    }

    /// Run the partitioning heuristic on a bare graph shape, returning
    /// `(colour, member indices)` per region in region-id order.
    fn run_heuristic(
        n_nodes: usize,
        edges: &[(usize, usize)],
        admissible: &[Vec<usize>],
        n_colours: usize,
    ) -> Vec<(usize, Vec<usize>)> {
        let mut graph: DiGraph<(), ()> = DiGraph::new();
        let nodes: Vec<NodeIndex> = (0..n_nodes).map(|_| graph.add_node(())).collect();
        for &(from, to) in edges {
            graph.add_edge(nodes[from], nodes[to], ());
        }
        let topo_order: Vec<NodeIndex> = lexicographical_topological_sort(
            &graph,
            |n: NodeIndex| Ok::<usize, std::convert::Infallible>(n.index()),
            false,
            None,
        )
        .unwrap();
        let assignment = assign_regions(
            &graph,
            &topo_order,
            admissible,
            &vec![false; n_nodes],
            n_colours,
        );
        assignment
            .members
            .iter()
            .enumerate()
            .map(|(rid, members)| {
                (
                    assignment.colour_of_region[rid],
                    members.iter().map(|v| v.index()).collect(),
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // The heuristic
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_program() {
        let contracted = QuantumProgram::new().into_regions(&[]).unwrap();
        assert!(contracted.regions.is_empty());
        assert!(contracted.program.call_flat(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_single_category_merges_whole_chain() {
        let regions = run_heuristic(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], &vec![vec![0]; 5], 2);
        assert_eq!(regions, vec![(0, vec![0, 1, 2, 3, 4])]);
    }

    #[test]
    fn test_alternating_categories_split() {
        let regions = run_heuristic(3, &[(0, 1), (1, 2)], &[vec![0], vec![1], vec![0]], 2);
        assert_eq!(regions, vec![(0, vec![0]), (1, vec![1]), (0, vec![2])]);
    }

    #[test]
    fn test_fallback_category_is_shared() {
        // `Store` and `Mean` are both unlisted, sit at the same depth, and so share the
        // single fallback region even though they are different node types.
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(4.0)).unwrap();
        prog.add_node("mean", Mean::new(0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("mean", vec![]))
            .unwrap();
        prog.add_edge(
            Port::new("mean", vec![]),
            Port::new("add", vec!["x".into()]),
        )
        .unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("out", Port::new("add", vec![])).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.add"]])).unwrap();
        assert_eq!(contracted.regions.len(), 2);
        assert_eq!(contracted.regions[0].category, None);
        assert_eq!(contracted.regions[0].members, ["s", "mean"]);
        assert_eq!(contracted.regions[1].category, Some(0));
        assert_eq!(contracted.regions[1].members, ["add"]);
    }

    #[test]
    fn test_diamond_not_over_merged() {
        // A(c0) -> B(c1) -> D(c0) and A -> C(c0) -> D. `C` joins `A`, but `D` cannot:
        // merging it would make the region re-enterable through `B`.
        let regions = run_heuristic(
            4,
            &[(0, 1), (1, 3), (0, 2), (2, 3)],
            &[vec![0], vec![1], vec![0], vec![0]],
            2,
        );
        assert_eq!(regions, vec![(0, vec![0, 2]), (1, vec![1]), (0, vec![3])]);
    }

    #[test]
    fn test_disconnected_same_category_merge() {
        // Intentional: nodes with no path between them still merge, so two independent
        // same-category nodes become one region (and hence one submission).
        let regions = run_heuristic(2, &[], &vec![vec![0]; 2], 2);
        assert_eq!(regions, vec![(0, vec![0, 1])]);
    }

    #[test]
    fn test_flexible_category_reduces_region_count() {
        // Node 1's type is listed in both category 0 and category 1. Taking category 0
        // (the earlier listing) would split it off from node 0; it takes category 1 to
        // keep the region count at one.
        let regions = run_heuristic(2, &[(0, 1)], &[vec![1], vec![0, 1]], 3);
        assert_eq!(regions, vec![(1, vec![0, 1])]);
    }

    #[test]
    fn test_flexible_category_tie_breaks_to_first_list() {
        let regions = run_heuristic(1, &[], &[vec![0, 1]], 3);
        assert_eq!(regions, vec![(0, vec![0])]);
    }

    #[test]
    fn test_regions_sorted_topologically() {
        let contracted = mixed_program().into_regions(&mixed_categories()).unwrap();
        let region_id = |label: &str| -> Option<usize> {
            label.strip_prefix("region_").map(|n| n.parse().unwrap())
        };
        let mut checked = 0;
        for (from, to) in contracted.program.iter_edges() {
            if let (Some(src), Some(tgt)) = (region_id(&from.label), region_id(&to.label)) {
                assert!(
                    src < tgt,
                    "edge {} -> {} is out of order",
                    from.label,
                    to.label
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "expected at least one inter-region edge");
    }

    #[test]
    fn test_deterministic_labels_and_members() {
        let render = |contracted: &ContractedProgram| {
            contracted
                .regions
                .iter()
                .map(|r| {
                    (
                        r.label.clone(),
                        r.category,
                        r.members.clone(),
                        region_keys(&r.input_ports).join(","),
                        region_keys(&r.output_ports).join(","),
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = mixed_program().into_regions(&mixed_categories()).unwrap();
        let second = mixed_program().into_regions(&mixed_categories()).unwrap();
        assert_eq!(render(&first), render(&second));
    }

    #[test]
    fn test_cycle_errors() {
        let mut prog = QuantumProgram::new();
        prog.add_node("a", Mean::new(0)).unwrap();
        prog.add_node("b", Mean::new(0)).unwrap();
        prog.add_edge(Port::new("a", vec![]), Port::new("b", vec![]))
            .unwrap();
        prog.add_edge(Port::new("b", vec![]), Port::new("a", vec![]))
            .unwrap();
        let Err(err) = prog.into_regions(&[]) else {
            panic!("expected a cycle error");
        };
        assert!(matches!(err, ContractionError::Cycle));
    }

    // -----------------------------------------------------------------------
    // `Input` hoisting
    // -----------------------------------------------------------------------

    #[test]
    fn test_hoists_input_nodes() {
        let mut prog = QuantumProgram::new();
        prog.add_node("add", Add).unwrap();
        prog.add_node("s", make_store(1.0)).unwrap();
        let x = prog.add_input("x", concrete_1d(1)).unwrap();
        prog.add_edge(x, Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("out", Port::new("add", vec![])).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.add"]])).unwrap();
        assert!(contracted.program.has_node("__input_x"));
        for region in &contracted.regions {
            assert!(!region.members.iter().any(|m| m == "__input_x"));
        }
    }

    #[test]
    fn test_input_fanout_into_single_region() {
        let mut prog = QuantumProgram::new();
        prog.add_node("add", Add).unwrap();
        let x = prog.add_input("x", concrete_1d(3)).unwrap();
        prog.add_edge(x.clone(), Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.add_edge(x, Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("result", Port::new("add", vec![])).unwrap();

        let args = [Tensor::from([1.0_f64, 2.0, 3.0])];
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.add"]])).unwrap();
        assert_eq!(contracted.regions.len(), 1);
        assert_eq!(
            region_keys(&contracted.regions[0].input_ports),
            ["add.x", "add.y"]
        );
        // The fan-out survives as two parallel outer edges.
        assert_eq!(contracted.program.iter_edges().count(), 2);
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    #[test]
    fn test_program_output_declared_on_input_node() {
        // An `add_input` port used *directly* as a program output must survive hoisting.
        let mut prog = QuantumProgram::new();
        prog.add_node("mean", Mean::new(0)).unwrap();
        let x = prog.add_input("x", concrete_1d(3)).unwrap();
        prog.add_edge(x.clone(), Port::new("mean", vec![])).unwrap();
        prog.set_output("passthru", x).unwrap();
        prog.set_output("avg", Port::new("mean", vec![])).unwrap();

        let args = [Tensor::from([1.0_f64, 2.0, 3.0])];
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.mean"]])).unwrap();
        assert_eq!(
            top_keys(contracted.program.output_types()),
            ["passthru", "avg"]
        );
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    #[test]
    fn test_unwired_input_node_preserved() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(2.0)).unwrap();
        prog.add_input("x", concrete_1d(1)).unwrap();
        prog.set_output("out", Port::new("s", vec![])).unwrap();

        let in_keys = top_keys(prog.input_types());
        let out_keys = top_keys(prog.output_types());
        let args = [Tensor::from([9.0_f64])];
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog.into_regions(&[]).unwrap();
        assert!(contracted.program.has_node("__input_x"));
        assert_eq!(top_keys(contracted.program.input_types()), in_keys);
        assert_eq!(top_keys(contracted.program.output_types()), out_keys);
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    // -----------------------------------------------------------------------
    // I/O preservation
    // -----------------------------------------------------------------------

    /// Two `set_input`s, two `add_input`s and three `set_output`s spread over three
    /// categories.
    fn io_order_program() -> QuantumProgram {
        let mut prog = QuantumProgram::new();
        prog.add_node("add", Add).unwrap();
        prog.add_node("mul", Multiply).unwrap();
        prog.add_node("mean", Mean::new(0)).unwrap();
        prog.set_input("a", Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.set_input("b", Port::new("add", vec!["y".into()]))
            .unwrap();
        let c = prog.add_input("c", concrete_1d(1)).unwrap();
        let d = prog.add_input("d", concrete_1d(1)).unwrap();
        prog.add_edge(c, Port::new("mul", vec!["x".into()]))
            .unwrap();
        prog.add_edge(d, Port::new("mul", vec!["y".into()]))
            .unwrap();
        prog.add_edge(Port::new("add", vec![]), Port::new("mean", vec![]))
            .unwrap();
        prog.set_output("o1", Port::new("add", vec![])).unwrap();
        prog.set_output("o2", Port::new("mean", vec![])).unwrap();
        prog.set_output("o3", Port::new("mul", vec![])).unwrap();
        prog
    }

    #[test]
    fn test_preserves_io_key_order() {
        let prog = io_order_program();
        let in_keys = top_keys(prog.input_types());
        let out_keys = top_keys(prog.output_types());
        assert_eq!(in_keys, ["a", "b", "c", "d"]);
        assert_eq!(out_keys, ["o1", "o2", "o3"]);

        let args: Vec<Tensor> = [1.0_f64, 2.0, 3.0, 4.0]
            .into_iter()
            .map(|v| Tensor::from([v]))
            .collect();
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog
            .into_regions(&cats(&[&["qiskit.add"], &["qiskit.mean"]]))
            .unwrap();
        assert_eq!(top_keys(contracted.program.input_types()), in_keys);
        assert_eq!(top_keys(contracted.program.output_types()), out_keys);
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    #[test]
    fn test_output_fanout_shared_source_port() {
        // `s`'s single output feeds two program outputs and one other region, but the
        // region exposes it under exactly one key.
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(4.0)).unwrap();
        prog.add_node("mean", Mean::new(0)).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("mean", vec![]))
            .unwrap();
        prog.set_output("a", Port::new("s", vec![])).unwrap();
        prog.set_output("b", Port::new("s", vec![])).unwrap();
        prog.set_output("c", Port::new("mean", vec![])).unwrap();

        let expected = prog.call_flat(&[]).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.mean"]])).unwrap();
        assert_eq!(contracted.regions.len(), 2);
        assert_eq!(region_keys(&contracted.regions[0].output_ports), ["s"]);
        assert_debug_eq(&contracted.program.call_flat(&[]).unwrap(), &expected);
    }

    #[test]
    fn test_same_key_on_both_sides() {
        // `Mean`'s input and output leaves are both at the root path, so its region keys
        // both sides to the bare member label. That's fine: a program's input and output
        // keys live in separate namespaces.
        let mut prog = QuantumProgram::new();
        prog.add_node(
            "s",
            Store::new(DataTree::new_leaf(Tensor::from([1.0_f64, 3.0]))),
        )
        .unwrap();
        prog.add_node("mean", Mean::new(0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("mean", vec![]))
            .unwrap();
        prog.add_edge(
            Port::new("mean", vec![]),
            Port::new("add", vec!["x".into()]),
        )
        .unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("out", Port::new("add", vec![])).unwrap();

        let expected = prog.call_flat(&[]).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.mean"]])).unwrap();
        let mean_region = contracted
            .regions
            .iter()
            .find(|r| r.members == ["mean"])
            .expect("`mean` gets a region of its own");
        assert_eq!(region_keys(&mean_region.input_ports), ["mean"]);
        assert_eq!(region_keys(&mean_region.output_ports), ["mean"]);
        assert_debug_eq(&contracted.program.call_flat(&[]).unwrap(), &expected);
    }

    #[test]
    fn test_port_key_collision_is_deduped() {
        // Node `a`'s input port `x` and node `a.x`'s root input port both flatten to the
        // string `"a.x"`; the second one gets a `#1` suffix instead of colliding.
        let mut prog = QuantumProgram::new();
        prog.add_node("a", Add).unwrap();
        prog.add_node("a.x", Mean::new(0)).unwrap();
        prog.set_input("i0", Port::new("a", vec!["x".into()]))
            .unwrap();
        prog.set_input("i1", Port::new("a", vec!["y".into()]))
            .unwrap();
        prog.set_input("i2", Port::new("a.x", vec![])).unwrap();
        prog.set_output("o0", Port::new("a", vec![])).unwrap();
        prog.set_output("o1", Port::new("a.x", vec![])).unwrap();

        let args = [
            Tensor::from([1.0_f64]),
            Tensor::from([2.0_f64]),
            Tensor::from([1.0_f64, 3.0]),
        ];
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog
            .into_regions(&cats(&[&["qiskit.add", "qiskit.mean"]]))
            .unwrap();
        assert_eq!(contracted.regions.len(), 1);
        assert_eq!(
            region_keys(&contracted.regions[0].input_ports),
            ["a.x", "a.y", "a.x#1"]
        );
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    // -----------------------------------------------------------------------
    // End-to-end equivalence
    // -----------------------------------------------------------------------

    /// A program mixing `Store`, `Add`, `Multiply` and `Mean` nodes, program inputs of
    /// both kinds, an input fan-out and two outputs sharing one source port.
    fn mixed_program() -> QuantumProgram {
        let mut prog = QuantumProgram::new();
        prog.add_node(
            "s1",
            Store::new(DataTree::new_leaf(Tensor::from([1.0_f64, 2.0]))),
        )
        .unwrap();
        prog.add_node("s2", make_store(10.0)).unwrap();
        prog.add_node("mul", Multiply).unwrap();
        prog.add_node("add1", Add).unwrap();
        prog.add_node("add2", Add).unwrap();
        prog.add_node("mean", Mean::new(0)).unwrap();
        prog.add_node("add3", Add).unwrap();
        prog.add_node("add4", Add).unwrap();

        prog.set_input("p", Port::new("mul", vec!["x".into()]))
            .unwrap();
        prog.set_input("q", Port::new("mul", vec!["y".into()]))
            .unwrap();
        let x = prog.add_input("x", concrete_1d(2)).unwrap();

        prog.add_edge(x.clone(), Port::new("add1", vec!["x".into()]))
            .unwrap();
        prog.add_edge(x, Port::new("add1", vec!["y".into()]))
            .unwrap();
        prog.add_edge(
            Port::new("add1", vec![]),
            Port::new("add2", vec!["x".into()]),
        )
        .unwrap();
        prog.add_edge(Port::new("s1", vec![]), Port::new("add2", vec!["y".into()]))
            .unwrap();
        prog.add_edge(Port::new("add2", vec![]), Port::new("mean", vec![]))
            .unwrap();
        prog.add_edge(
            Port::new("mean", vec![]),
            Port::new("add3", vec!["x".into()]),
        )
        .unwrap();
        prog.add_edge(Port::new("s2", vec![]), Port::new("add3", vec!["y".into()]))
            .unwrap();
        prog.add_edge(
            Port::new("mul", vec![]),
            Port::new("add4", vec!["x".into()]),
        )
        .unwrap();
        prog.add_edge(
            Port::new("add3", vec![]),
            Port::new("add4", vec!["y".into()]),
        )
        .unwrap();

        prog.set_output("o1", Port::new("add2", vec![])).unwrap();
        prog.set_output("o2", Port::new("mean", vec![])).unwrap();
        prog.set_output("o3", Port::new("add2", vec![])).unwrap();
        prog.set_output("o4", Port::new("add4", vec![])).unwrap();
        prog
    }

    fn mixed_args() -> Vec<Tensor> {
        vec![
            Tensor::from([3.0_f64]),
            Tensor::from([4.0_f64]),
            Tensor::from([1.0_f64, 2.0]),
        ]
    }

    fn mixed_categories() -> Vec<Vec<String>> {
        cats(&[&["qiskit.add"], &["qiskit.mean"]])
    }

    #[test]
    fn test_call_flat_equivalence() {
        let prog = mixed_program();
        let args = mixed_args();
        let in_keys = top_keys(prog.input_types());
        let out_keys = top_keys(prog.output_types());
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog.into_regions(&mixed_categories()).unwrap();
        assert!(contracted.regions.len() > 1, "expected a real partition");
        assert_eq!(top_keys(contracted.program.input_types()), in_keys);
        assert_eq!(top_keys(contracted.program.output_types()), out_keys);
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    #[test]
    fn test_resolve_types_flat_equivalence() {
        let prog = mixed_program();
        let arg_types = vec![concrete_1d(1), concrete_1d(1), concrete_1d(2)];
        let expected = prog.resolve_types_flat(&arg_types).unwrap();

        let contracted = prog.into_regions(&mixed_categories()).unwrap();
        assert_debug_eq(
            &contracted.program.resolve_types_flat(&arg_types).unwrap(),
            &expected,
        );
    }

    #[test]
    fn test_call_flat_equivalence_with_nested_program() {
        let mut inner = QuantumProgram::new();
        inner.add_node("s", make_store(1.0)).unwrap();
        inner.add_node("add", Add).unwrap();
        inner
            .add_edge(Port::new("s", vec![]), Port::new("add", vec!["x".into()]))
            .unwrap();
        inner
            .set_input("y", Port::new("add", vec!["y".into()]))
            .unwrap();
        inner.set_output("sum", Port::new("add", vec![])).unwrap();

        let mut prog = QuantumProgram::new();
        prog.add_node("nested", inner).unwrap();
        prog.add_node("mean", Mean::new(0)).unwrap();
        prog.add_node("mul", Multiply).unwrap();
        let x = prog.add_input("x", concrete_1d(2)).unwrap();
        prog.add_edge(x.clone(), Port::new("nested", vec!["y".into()]))
            .unwrap();
        prog.add_edge(
            Port::new("nested", vec!["sum".into()]),
            Port::new("mean", vec![]),
        )
        .unwrap();
        prog.add_edge(
            Port::new("mean", vec![]),
            Port::new("mul", vec!["x".into()]),
        )
        .unwrap();
        prog.add_edge(x, Port::new("mul", vec!["y".into()]))
            .unwrap();
        prog.set_output("out", Port::new("mul", vec![])).unwrap();

        let args = [Tensor::from([1.0_f64, 2.0])];
        let expected = prog.call_flat(&args).unwrap();

        let contracted = prog
            .into_regions(&cats(&[&["qiskit.quantum_program"], &["qiskit.mean"]]))
            .unwrap();
        assert_debug_eq(&contracted.program.call_flat(&args).unwrap(), &expected);
    }

    #[test]
    fn test_contraction_is_idempotent() {
        let args = mixed_args();
        let expected = mixed_program().call_flat(&args).unwrap();

        let once = mixed_program().into_regions(&mixed_categories()).unwrap();
        let twice = once
            .program
            .into_regions(&cats(&[&["qiskit.quantum_program"]]))
            .unwrap();
        // Every region node has the same type now, so they all merge back together.
        assert_eq!(twice.regions.len(), 1);
        assert_eq!(twice.regions[0].category, Some(0));
        assert_debug_eq(&twice.program.call_flat(&args).unwrap(), &expected);
    }

    // -----------------------------------------------------------------------
    // Region access and documented deviations
    // -----------------------------------------------------------------------

    #[test]
    fn test_as_quantum_program_reaches_region() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(1.0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("out", Port::new("add", vec![])).unwrap();

        let contracted = prog.into_regions(&cats(&[&["qiskit.add"]])).unwrap();
        for region in &contracted.regions {
            let node = contracted.program.get_node(&region.label).unwrap();
            let inner = node
                .as_quantum_program()
                .expect("a region node is a QuantumProgram");
            let mut labels: Vec<&str> = inner.iter_nodes().map(|(label, _)| label).collect();
            labels.sort_unstable();
            let mut members = region.members.clone();
            members.sort();
            assert_eq!(labels, members);
        }
        // Ordinary nodes are not programs.
        let store_region = contracted.program.get_node("region_0").unwrap();
        assert!(
            store_region
                .as_quantum_program()
                .unwrap()
                .get_node("s")
                .unwrap()
                .as_quantum_program()
                .is_none()
        );
    }

    #[test]
    fn test_unwired_member_input_nests_error() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(1.0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.set_output("out", Port::new("add", vec![])).unwrap();

        // Contraction itself doesn't validate wiring.
        let contracted = prog.into_regions(&cats(&[&["qiskit.add"]])).unwrap();
        let err = contracted.program.call_flat(&[]).unwrap_err();
        let QuantumProgramCallError::NodeCall { label, source } = err else {
            panic!("expected a NodeCall error, got {err:?}");
        };
        assert_eq!(label, "region_1");
        let nested = source
            .downcast_ref::<QuantumProgramCallError>()
            .expect("the region's own error is nested inside");
        assert!(matches!(
            nested,
            QuantumProgramCallError::UnwiredInput { label, .. } if label == "add"
        ));
    }

    #[test]
    fn test_output_less_region_is_pruned() {
        // `a` cannot be called, and the original program surfaces that because `b`
        // consumes its output. Contracted, `{a, b}` is one region with no outputs at all,
        // which `run_topo` skips wholesale — so the error disappears.
        let mut prog = QuantumProgram::new();
        prog.add_node("a", Input::new(concrete_1d(2))).unwrap();
        prog.add_node("b", Mean::new(0)).unwrap();
        prog.add_edge(Port::new("a", vec![]), Port::new("b", vec![]))
            .unwrap();

        let err = prog.call_flat(&[]).unwrap_err();
        assert!(matches!(
            err,
            QuantumProgramCallError::NodeCall { ref label, .. } if label == "a"
        ));

        let contracted = prog
            .into_regions(&cats(&[&["qiskit.input", "qiskit.mean"]]))
            .unwrap();
        assert_eq!(contracted.regions.len(), 1);
        assert!(contracted.regions[0].output_ports.is_empty());
        assert!(contracted.program.call_flat(&[]).unwrap().is_empty());
    }
}
