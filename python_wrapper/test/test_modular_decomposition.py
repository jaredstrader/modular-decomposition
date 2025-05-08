# tests/test_modules.py

import random
import logging
import pytest
import networkx as nx
from itertools import combinations
from rust_mod_decomp import modular_decompose, module_neighbors

def make_random_graph(n, p=0.5):
    """
    Build Erdos-Renyi graph in networkx
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)
    return G

def is_module(G: nx.Graph, X):
    """
    X is a module iff every outside node is adjacent to either
    all of X or to none of X.
    """
    X = set(X)
    if not X:
        return False

    v = next(iter(X))
    required = set(G.neighbors(v)) - X
    for u in X:
        if set(G.neighbors(u)) - X != required:
            return False
    return True

def find_all_modules_brute_force(G: nx.Graph):
    """
    Return the strong modules of G by brute force.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    #all candidate modules with size greater than two
    all_mods = [tuple(sorted(c)) 
                for k in range(2, n + 1)
                for c in combinations(nodes, k)
                if is_module(G, c)]

    #all strong modules (e.g., no partial overlap)
    strong = []
    for M in all_mods:
        Mset = set(M)
        ok = True
        for N in all_mods:
            if M == N:
                continue
            inter = Mset & set(N)
            if inter and inter != Mset and inter != set(N):
                ok = False
                break
        if ok:
            strong.append(M)
    return sorted(strong)

def brute_module_neighbors(G: nx.Graph, modules):
    """
    Return the sorted list of neighborhoods of each module.
    """
    out = []
    for M in modules:
        Mset = set(M)
        neigh = set()
        for u in M:
            for v in G.neighbors(u):
                if v not in Mset:
                    neigh.add(v)
        out.append(tuple(sorted(neigh)))
    return out

@pytest.mark.parametrize("n,p,trials", [(5, 0.75, 20), (10, 0.75, 20)])
def test_random_graphs(n, p, trials):
    for _ in range(trials):
        G = make_random_graph(n, p=p)
        adj = [list(G.neighbors(i)) for i in range(n)]

        _, rust_mods = modular_decompose(adj, backend="linear")
        rust_neigh   = module_neighbors(adj, rust_mods)

        true_mods    = find_all_modules_brute_force(G)
        true_neigh   = [list(nb) for nb in brute_module_neighbors(G, true_mods)]

        logging.info(
            f"G: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
            f"mods={len(rust_mods)}, nbrs={[len(x) for x in rust_neigh]}"
        )

        assert set(map(tuple, rust_mods)) == set(true_mods)
        for M in rust_mods:
            idx_r = rust_mods.index(M)
            idx_t = true_mods.index(tuple(sorted(M)))
            assert rust_neigh[idx_r] == true_neigh[idx_t]
