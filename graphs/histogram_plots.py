import tabulate
import numpy as np
import matplotlib.pyplot as plt

from eval import (
    evaluate,
    plot_clustering_coefficient_histogram,
    plot_eigenvector_centrality,
    plot_node_degree_histogram,
)


data = [
    ['MUTAG', 'MUTAG.npy'],
    ['Erdös-Rényi', 'ERDOS.npy'],
    ['GNN MP', 'chris_mm.npy'],
    ['GNN GC', 'chris_conv.npy'],
    ['GNN_node MP', 'mp_node.npy'],
    ['GNN_node GC', 'mp_conv.npy'],
]

for i, (name, file) in enumerate(data):
    _path = f'./samples/{file}'
    data[i][1] = np.load(_path)

kwargs = dict(
    density=True,
    bins=10,
    alpha=0.5,
)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

d = [data[i] for i in (0, 1, 2, 3)]
plot_clustering_coefficient_histogram(*d, ax=axs[0, 0], **kwargs )
plot_eigenvector_centrality(*d, ax=axs[0, 1], **kwargs )
plot_node_degree_histogram(*d, ax=axs[0, 2], **kwargs )

d = [data[i] for i in (0, 1, 4, 5)]
plot_clustering_coefficient_histogram(*d, ax=axs[1, 0], **kwargs )
plot_eigenvector_centrality(*d, ax=axs[1, 1], **kwargs )
plot_node_degree_histogram(*d, ax=axs[1, 2], **kwargs )

fig.tight_layout()
fig.savefig('graphs.pdf')
plt.show()


tbl_data = list()
for name, adj in data:
    tbl_data.append([name, *[f'{100*e:.1f}%' for e in evaluate(adj)]])

headers = ['', 'Novel', 'Unique', 'Novel+Unique']
table = tabulate.tabulate(tbl_data, headers=headers, tablefmt='latex_raw')
print(table)