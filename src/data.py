import os
import pickle
import random
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
from src.models.GraphRNN.utils import graph_to_sequence, compute_max_prev_node


#Synthetic/real graph generators
def generate_community_graphs(num_graphs=500, min_nodes=60, max_nodes=160, p_intra=0.3, inter_edge_ratio=0.05, seed=0):
    rng=random.Random(seed)
    graphs=[]
    for i in range(num_graphs):
        n=rng.randint(min_nodes, max_nodes)
        n1=n//2
        n2=n-n1
        c1=nx.erdos_renyi_graph(n1, p_intra, seed=rng.randint(0, 10**6))
        c2=nx.erdos_renyi_graph(n2, p_intra, seed=rng.randint(0, 10**6))
        c2=nx.relabel_nodes(c2, {k:k+n1 for k in c2.nodes()})
        G=nx.union(c1, c2)
        num_inter=max(1, int(inter_edge_ratio*n))
        nodes1, nodes2=list(c1.nodes()), list(c2.nodes())

        for _ in range(num_inter):
            G.add_edge(rng.choice(nodes1), rng.choice(nodes2))
        if not nx.is_connected(G):
            largest=max(nx.connected_components(G), key=len)
            G=G.subgraph(largest).copy()
        graphs.append(nx.convert_node_labels_to_integers(G))
    return graphs


def generate_grid_graphs(num_graphs=100, min_side=10, max_side=20, seed=0):
    rng=random.Random(seed)
    graphs=[]
    for _ in range(num_graphs):
        m=rng.randint(min_side, max_side)
        n=rng.randint(min_side, max_side)
        G=nx.grid_2d_graph(m,n)
        graphs.append(nx.convert_node_labels_to_integers(G))
    return graphs


def generate_ego_graphs(num_graphs=500, radius=3, min_nodes=50, max_nodes=399, root="data/raw", seed=0):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx

    dataset=Planetoid(root=root, name="Citeseer")
    data=dataset[0]
    G_full=to_networkx(data, to_undirected=True)

    nodes=list(G_full.nodes())
    random.Random(seed).shuffle(nodes)

    graphs=[]
    for n in nodes:
        if len(graphs) >= num_graphs:
            break
        ego=nx.ego_graph(G_full, n, radius=radius)
        if min_nodes <= ego.number_of_nodes() <= max_nodes:
            graphs.append(nx.convert_node_labels_to_integers(ego))
    return graphs

DATASET_BUILDERS = {
    "community": generate_community_graphs,
    "grid": generate_grid_graphs,
    "ego": generate_ego_graphs
}


#Dataset loading
def get_dataset(name, data_dir="data", force_reload=False, seed=0):
    processed_path=os.path.join(data_dir, "processed", f"{name}.pkl")

    if os.path.exists(processed_path) and not force_reload:
        with open(processed_path, "rb") as f:
            graphs=pickle.load(f)
    else:
        if name=="ego":
            graphs=generate_ego_graphs(root=os.path.join(data_dir, "raw"), seed=seed)
        else:
            graphs=DATASET_BUILDERS[name](seed=seed)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        with open(processed_path, "wb") as f:
            pickle.dump(graphs, f)

    graphs=[G for G in graphs if G.number_of_nodes() > 1]
    random.Random(seed).shuffle(graphs)
    split=max(1, int(0.8 * len(graphs)))
    train_graphs, test_graphs=graphs[:split], graphs[split:]

    max_num_node=max(G.number_of_nodes() for G in graphs)
    max_prev_node=compute_max_prev_node(train_graphs)
    return train_graphs, test_graphs, max_num_node, max_prev_node