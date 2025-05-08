import networkx as nx
import matplotlib.pyplot as plt
import distinctipy
from rust_mod_decomp import modular_decompose, module_neighbors

#ontology
W = [
    "hallway", "family", "living", "full_bath", "half_bath", "kitchen"
]
B = [
    "wall", "floor", "couch", "tv", "bookshelf",
    "toilet", "sink", "fridge", "shower", "towel", "mirror"
]
edges = [
    ("hallway", "wall"),
    ("hallway", "floor"),
    ("family", "wall"),
    ("family", "floor"),
    ("family", "couch"),
    ("family", "tv"),
    ("family", "bookshelf"),
    ("living", "wall"),
    ("living", "floor"),
    ("living", "couch"),
    ("living", "bookshelf"),
    ("full_bath", "sink"),
    ("full_bath", "toilet"),
    ("full_bath", "wall"),
    ("full_bath", "floor"),
    ("full_bath", "shower"),
    ("full_bath", "towel"),
    ("full_bath", "mirror"),
    ("half_bath", "sink"),
    ("half_bath", "toilet"),
    ("half_bath", "wall"),
    ("half_bath", "floor"),
    ("half_bath", "towel"),
    ("half_bath", "mirror"),
    ("kitchen", "sink"),
    ("kitchen", "fridge"),
    ("kitchen", "wall"),
    ("kitchen", "floor"),
    ("kitchen", "towel"),
]

name2idx = {name: i for i, name in enumerate(W + B)}
idx2name = {i: name for name, i in name2idx.items()}

G = nx.Graph()
G.add_nodes_from(W, part=0)
G.add_nodes_from(B, part=1)
G.add_edges_from(edges)

n = len(name2idx)
adj = [[] for _ in range(n)]
for u, v in G.edges():
    uid, vid = name2idx[u], name2idx[v]
    adj[uid].append(vid)
    adj[vid].append(uid)  #make undirected

print('adjacency list:', adj)

for i, nbrs in enumerate(adj):
    for j in nbrs:
        assert j != i, f"Self-loop at node {i}"
        assert 0 <= j < len(adj), f"Invalid edge: ({i}, {j})"

#plot graph
positions = {}
colors = distinctipy.get_colors(n, rng=3)
node_colors = {}

region_names = W
mesh_names = B
max_len = max(len(region_names), len(mesh_names))
offset_w = (max_len - len(region_names)) / 2
offset_b = (max_len - len(mesh_names)) / 2

for i, name in enumerate(region_names):
    positions[name] = (2, i + offset_w)
    node_colors[name] = colors[name2idx[name]]

for i, name in enumerate(mesh_names):
    positions[name] = (0, i + offset_b)
    node_colors[name] = colors[name2idx[name]]

plt.figure(figsize=(7, 5))
nx.draw(
    G,
    pos=positions,
    with_labels=True,
    node_color=[node_colors[n] for n in G.nodes()],
    node_size=1000,
    edge_color="gray",
    font_weight="bold",
)
plt.title("Bipartite Ontology Graph with Modular Decomposition Nodes")
plt.axis("off")
plt.tight_layout()
plt.show()


#modular decomposition
labels, modules = modular_decompose(adj, backend="skeleton")

print("\nModular decomposition labels (node_index, kind):")
for node_id, kind in labels:
    label = idx2name.get(node_id, f"<internal_{node_id}>")
    print(f"Node {node_id:>2} ({label:<14}): {kind}")

print("\nExtracted modules (sets of vertex indices):")
for i, module in enumerate(modules):
    names = [idx2name[idx] for idx in module]
    print(f"Module {i + 1:>2}: {names}")

#neighborhoods
neighbors = module_neighbors(adj, modules)

print("\nModule neighbors (module_index, neighbor_indices):")
for i, module in enumerate(modules):
    neighbors_list = neighbors[i]
    names = [idx2name[idx] for idx in neighbors_list]
    print(f"Module {i + 1:>2}: {names}")