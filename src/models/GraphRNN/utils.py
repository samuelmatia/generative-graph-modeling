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
    for i in range(1,n):
        row=adj[i, :i][::-1]
        row=row[:max_prev_node]
        S[i-1, :len(row)] = row
    return S
    

#Inverse of encode_adj, reconstruct (n,n) adjacency matrix from S
def decode_adj(S):
    n=S.shape[0]+1
    M=S.shape[1]
    adj=np.zeros((n,n), dtype=float32)
    for i in range(1, n):
        row=S[i-1]
        num_prev=min(i,M)
        for k in range(num_prev):
            j=i-1-k
            adj[i, j]=row[k]
    return ajd+adj.T  #undirected graph


def graph_to_sequence(G, max_prev_node, max_num_node=None):
    order=get_bfs_order(G)
    if max_num_node is not None:
        order=order[:max_num_node]
    sub=G.subgraph(order)
    adj=nx.to_numpy_array(sub, nodelist=order)
    return encode_adj(adj, max_prev_node)


def sequence_to_graph(S):
    adj=decode_adj(S)
    G=nx.from_numpy_array(adj)
    #G.remove_nodes_from(list(nx.isolates(G)))
    return G


def compute_max_prev_node(graphs, num_samples=1000, percentile=99.9, seed=0):
    rng=random.Random(seed)
    gaps=[]
    for _ in range(num_samples):
        G=rng.choice(graphs)
        order=get_bfs_order(G)
        idx= {node:i for i,node in enumerate(order)}

        for u,v in G.edges():
            if u in idx and v in idx:
                gaps.append(abs(idx[u]-idx[v]))
    if not gaps:
        return 1
    return int(np.percentile(gaps, percentile)) + 1


def make_mask(lengths, T, M, device=None):
    B=len(lengths)
    mask=np.zeros((B, T, M), dtype=np.float32)
    for b, L in enumerate(lengths):
        L=int(L)
        for t in range(L):
            valid=min(t+1, M)
            mask[b, t, :valid] = 1.0
    import torch
    m=torch.from_numpy(mask)
    return m.to(device) if device is not None else m