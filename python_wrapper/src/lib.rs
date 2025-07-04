use pyo3::prelude::*;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::Dfs;
use std::collections::HashSet;
use common::modular_decomposition::MDNodeKind;
use linear::modular_decomposition as linear_modular_decomposition;

//build an undirected graph from adjacency lists
fn from_adjacency_list(adj: Vec<Vec<usize>>) -> UnGraph<(), ()> {
    let mut g = UnGraph::new_undirected();
    for _ in 0..adj.len() {
        g.add_node(());
    }
    for (u, neighbors) in adj.iter().enumerate() {
        for &v in neighbors {
            g.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
        }
    }
    g
}

//flatten the mdtree
fn format_labels(tree: &DiGraph<MDNodeKind, ()>) -> Vec<(usize, String)> {
    tree.node_indices()
        .map(|i| (i.index(), format!("{:?}", tree[i].clone())))
        .collect()
}

//for each non‐leaf in the mdtree, run a dfs to collect all descendant
//vertices to get the modules
fn extract_modules(tree: &DiGraph<MDNodeKind, ()>) -> Vec<Vec<usize>> {
    let mut modules = Vec::new();
    for node in tree.node_indices() {
        match tree[node] {
            MDNodeKind::Vertex(_) => continue,  // skip leaves
            _ => {
                let mut dfs = Dfs::new(tree, node);
                let mut module = Vec::new();
                while let Some(nx) = dfs.next(tree) {
                    if let MDNodeKind::Vertex(v) = tree[nx] {
                        module.push(v);
                    }
                }
                module.sort_unstable();
                modules.push(module);
            }
        }
    }
    modules
}

//py fn to get neighborhoods of modules
#[pyfunction]
fn module_neighbors(
    adj: Vec<Vec<usize>>,
    modules: Vec<Vec<usize>>,
) -> PyResult<Vec<Vec<usize>>> {
    // rebuild the graph
    let g = from_adjacency_list(adj);

    // for each module, collect neighbor set
    let mut all_neighbors = Vec::with_capacity(modules.len());
    for module in modules {
        let module_set: HashSet<usize> = module.iter().cloned().collect();
        let mut neigh = HashSet::new();

        for &v in &module {
            let idx = NodeIndex::new(v);
            for nbr in g.neighbors(idx) {
                let u = nbr.index();
                if !module_set.contains(&u) {
                    neigh.insert(u);
                }
            }
        }

        // sort and push
        let mut neigh_list: Vec<usize> = neigh.into_iter().collect();
        neigh_list.sort_unstable();
        all_neighbors.push(neigh_list);
    }

    Ok(all_neighbors)
}

//py fn to run modular decomposition
#[pyfunction]
fn modular_decompose(
    adj: Vec<Vec<usize>>,
    backend: Option<&str>,
) -> PyResult<(Vec<(usize, String)>, Vec<Vec<usize>>)> {
    let g = from_adjacency_list(adj);
    let tree: DiGraph<MDNodeKind, ()> = match backend {
        Some("fracture") => {
            std::panic::catch_unwind(|| fracture::modular_decomposition(&g))
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                    "fracture backend panicked during modular decomposition"
                ))?
        }
        Some("skeleton") => {
            std::panic::catch_unwind(|| skeleton::modular_decomposition(&g))
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                    "skeleton backend panicked during modular decomposition"
                ))?
        }
        Some("linear") => {
            std::panic::catch_unwind(|| linear_modular_decomposition(&g))
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                    "linear backend panicked during modular decomposition"
                ))?
        }
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown backend: {}. Use 'fracture', 'skeleton', or 'linear'.", other)
            ));
        }
        None => {
            std::panic::catch_unwind(|| linear_modular_decomposition(&g))
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                    "linear backend panicked during modular decomposition"
                ))?
        }
    };

    let labels  = format_labels(&tree);
    let modules = extract_modules(&tree);
    Ok((labels, modules))
}

//python module to expose rust interface
#[pymodule]
fn rust_mod_decomp(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__doc__", "Python bindings for modular decomposition using Rust backend.")?;
    m.add_function(wrap_pyfunction!(modular_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(module_neighbors, m)?)?;
    Ok(())
}
