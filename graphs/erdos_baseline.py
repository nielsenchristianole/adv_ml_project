# %%

from collections import defaultdict 
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

dataset = TUDataset(root='./data/', name='MUTAG')

# %%

graph_size_prob = defaultdict(lambda: 0)
graph_link_prob = defaultdict(lambda: 0)

def init_probs(dataset):

    for graph in dataset:       
        graph_size_prob[graph.num_nodes] += 1
        graph_link_prob[graph.num_nodes] += graph.num_edges

    for i, r in graph_link_prob.items():
        graph_link_prob[i] = r / (graph_size_prob[i] * i**2)

    num_graphs = sum(graph_size_prob.values())
    for i, r in graph_size_prob.items():
        graph_size_prob[i] = r / num_graphs

    return graph_size_prob, graph_link_prob


# %%
class ErdosRModel:
    def __init__(self, dataset):
        self.dataset = dataset

        self.graph_size_prob, self.graph_link_prob = init_probs(dataset)

    def generate(self, batch_size=1, adj_matrix=True):
        graphs = []
        ns = random.choices(list(self.graph_size_prob.keys()), weights=self.graph_size_prob.values(), k=batch_size)

        for n in ns:
            # num_edges = np.random.binomial(n=n**2, p=self.graph_link_prob[n])
            graph = nx.binomial_graph(n=n, p=self.graph_link_prob[n])
            # graphs.append(graph.adj)
            graphs.append(nx.adjacency_matrix(graph))

        return graphs

# %%

erdos_model = ErdosRModel(dataset=dataset)

if __name__ == '__main__':

    # Plot histogram
    plt.bar(graph_size_prob.keys(), graph_size_prob.values(), color='skyblue')
    plt.xlabel('Graph Size')
    plt.ylabel('Probability')
    plt.title('Histogram of Graph Size Probabilities')
    plt.show()

    # print(erdos_model)
    print(erdos_model.generate(batch_size=2)[0].todense())
