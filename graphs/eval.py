from typing import Optional
from functools import cache

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def graph_from_numpy(adjacency_matrix: np.ndarray) -> nx.Graph:
    """Converts an adjacency matrix to a networkx graph."""
    gr = nx.Graph()
    edges = zip(*np.where(adjacency_matrix))
    gr.add_edges_from(edges)
    return gr


def hash_adjacency_matrix(adjacency_matrix: np.ndarray) -> str:
    """Hashes an adjacency matrix."""

    gr = nx.Graph()
    edges = zip(*np.where(adjacency_matrix))
    gr.add_edges_from(edges)
    
    _hash = nx.weisfeiler_lehman_graph_hash(gr, iterations=10)
    return int(_hash, 16)


hash_adjacency_matrix_batch = np.vectorize(hash_adjacency_matrix, otypes=[object], signature='(k,k)->()')
graph_from_numpy_batch = np.vectorize(graph_from_numpy, otypes=[object], signature='(k,k)->()')


@cache
def get_dataset_adjacency_matrix() -> np.ndarray:
    """Get the adjacency matrix of the MUTAG dataset."""
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_adj

    dataset = TUDataset(root='./data/', name='MUTAG')
    dataloader = DataLoader(dataset, batch_size=100)

    adjacency_matrix = list()
    for data in dataloader:
        A = to_dense_adj(data.edge_index, data.batch)
        adjs = A.detach().cpu().numpy().astype(bool)
        adjacency_matrix.append(adjs)
    adjacency_matrix = np.concatenate(adjacency_matrix)

    assert adjacency_matrix.shape == (188, 28, 28)
    return adjacency_matrix


def evaluate(new_adjacency_matrix: np.ndarray) -> tuple[float, float, float]:
    """Evaluate the graph represented by the adjacency matrix."""
    assert len(new_adjacency_matrix.shape) == 3
    assert len(new_adjacency_matrix) == 1000 or len(new_adjacency_matrix) == 188

    MUTAG_adjacency_matrix = get_dataset_adjacency_matrix()
    MUTAG_hashes = hash_adjacency_matrix_batch(MUTAG_adjacency_matrix)
    MUTAG_set = set(MUTAG_hashes)

    new_hashes = hash_adjacency_matrix_batch(new_adjacency_matrix)
    new_set = set(new_hashes)

    _novel = np.mean([h not in MUTAG_set for h in new_hashes])
    _unique = len(new_set) / len(new_hashes)
    _novel_unique = len(new_set - MUTAG_set) / len(new_hashes)

    return (
        _novel,
        _unique,
        _novel_unique
    )


def plot_node_degree_histogram(*data: tuple[str, np.ndarray], ax: Optional[plt.Axes]=None, set_legent: bool=True, mean_over_graph: bool=False, colors=None, **kwargs):
    """Give a list of tuples (label, adjacency_matrix), plot the node degree histogram."""

    if ax is None:
        ax = plt.gca()

    kwargs = dict(
        density=True
    ) | kwargs

    if colors is None:
        colors = [None]*len(data)

    for (label, adjacency_matrix), color in zip(data, colors):
        assert len(adjacency_matrix.shape) == 3
        graphs = graph_from_numpy_batch(adjacency_matrix)
        if mean_over_graph:
            degrees = [np.mean([d for n, d in gr.degree()]) for gr in graphs]
        else:
            degrees = [d for gr in graphs for n, d in gr.degree()]
        ax.hist(degrees, label=label, color=color, **kwargs)

    if set_legent:
        ax.legend()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Node degree{"*" if mean_over_graph else ""}')

    return ax


def plot_clustering_coefficient_histogram(*data: tuple[str, np.ndarray], ax: Optional[plt.Axes]=None, set_legent: bool=True, mean_over_graph: bool=False, colors=None, **kwargs):
    """Give a list of tuples (label, adjacency_matrix), plot the node degree histogram."""

    if ax is None:
        ax = plt.gca()

    kwargs = dict(
        density=True
    ) | kwargs

    if colors is None:
        colors = [None]*len(data)

    for (label, adjacency_matrix), color in zip(data, colors):
        assert len(adjacency_matrix.shape) == 3
        graphs = graph_from_numpy_batch(adjacency_matrix)
        if mean_over_graph:
            clustering_coefficients = [nx.average_clustering(gr) for gr in graphs]
        else:
            clustering_coefficients = [nx.clustering(gr) for gr in graphs]
            clustering_coefficients = np.concatenate([np.array(list(d.values())) for d in clustering_coefficients])
        ax.hist(clustering_coefficients, label=label, color=color, **kwargs)

    if set_legent:
        ax.legend()
        ax.set_xlabel('Clustering coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Clustering coefficient{"*" if mean_over_graph else ""}')

    return ax


def calculate_eigenvector_centrality(graph: nx.Graph) -> dict[int, float]:
    """Calculate the eigenvector centrality of a graph."""
    for pow in range(1, 6):
        try:
            return nx.eigenvector_centrality(graph, max_iter=10**pow)
        except nx.PowerIterationFailedConvergence:
            pass
    raise nx.PowerIterationFailedConvergence


def plot_eigenvector_centrality(*data: tuple[str, np.ndarray], ax: Optional[plt.Axes]=None, set_legent: bool=True, mean_over_graph: bool=False, colors=None, **kwargs):
    """Give a list of tuples (label, adjacency_matrix), plot the node degree histogram."""

    if ax is None:
        ax = plt.gca()

    kwargs = dict(
        density=True
    ) | kwargs

    if colors is None:
        colors = [None]*len(data)

    for (label, adjacency_matrix), color in zip(data, colors):
        assert len(adjacency_matrix.shape) == 3
        graphs = graph_from_numpy_batch(adjacency_matrix)
        eigenvector_centralities = [calculate_eigenvector_centrality(gr) for gr in graphs]
        if mean_over_graph:
            eigenvector_centralities = [np.array(list(d.values())).mean() for d in eigenvector_centralities]
        else:
            eigenvector_centralities = np.concatenate([np.array(list(d.values())) for d in eigenvector_centralities])
        ax.hist(eigenvector_centralities, label=label, color=color, **kwargs)

    if set_legent:
        ax.legend()
        ax.set_xlabel('Eigenvector centrality')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Eigenvector centrality{"*" if mean_over_graph else ""}')

    return ax


if __name__ == '__main__':

    train_data = get_dataset_adjacency_matrix()

    plot_eigenvector_centrality(('MUTAG', train_data))
    plt.show()
    plot_clustering_coefficient_histogram(('MUTAG', train_data))
    plt.show()
    plot_node_degree_histogram(('MUTAG', train_data))
    plt.show()

    # print(evaluate(train_data))

