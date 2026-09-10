import random
import numpy as np
import networkx as nx

#return all the nodes in Breadth-First Search order
def get_bfs_order(G, start=None):
    nodes=list(G.nodes())
    if start is None:
        start=random.choice(nodes)
    order=[start]+[v for _, v in nx.bfs_edges(G,start)]
    for n in nodes:
        if n not in order:
            order.append(n)
    return order

# Encode the BFS-ordered adjacency matrix into a sequential representation
def encode_adj(adj, max_prev_node):
    n=adj.shape[0]
    adj=np.tril(adj, k=-1)
    S=np.zeros((n-1,max_prev_node), dtype=np.float32)
    

def decode_adj(S):
    pass


def graph_to_sequence():
    pass


def sequence_to_graph():
    pass


def compute_max_prev_node():
    pass


def make_mask():
    pass


