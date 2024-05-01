import pdb
import random
from itertools import permutations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.utils.data
from erdos_baseline import erdos_model
from eval import (
    evaluate,
    get_dataset_adjacency_matrix,
    plot_clustering_coefficient_histogram,
    plot_eigenvector_centrality,
    plot_node_degree_histogram,
)
from graph_vae_node import (
    BernoulliNodeDecoder,
    GaussianEncoderNodeConvolution,
    GaussianEncoderNodeMessagePassing,
    VAENode,
)
from scipy import stats
from scipy.stats import kde
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
from tqdm import tqdm

MUTAG_adjcs = get_dataset_adjacency_matrix()

# this is a function for sampling the number of nodes in a graph
global SAMPLES_GRAPH_SIZE_FROM_DATA
size, count = np.unique((MUTAG_adjcs.sum(axis=1) >= 1).sum(axis=1), return_counts=True)
SAMPLES_GRAPH_SIZE_FROM_DATA = lambda num_samples: np.random.choice(size, p=count/count.sum(), size=num_samples)


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoderMessagePassing(nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, M, dropout: float=0.1):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        -----------
            * node_feature_dim: the dimension of the node features
            * state_dim: dimensionality of the states
            * num_message_passing_rounds: you guessed it
            * M: encoding_dim (of the graph)

        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoderMessagePassing, self).__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim # 7
        self.state_dim = state_dim # 16
        self.num_message_passing_rounds = num_message_passing_rounds # 4

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
            ) for _ in range(num_message_passing_rounds)])
        
        # State output network
        self.encoder_net = torch.nn.Linear(self.state_dim, 2*M)

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        #state = self.input_net(x)
        state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            # For all edges between to nodes
            # we add the message of the from-node to the to-node.
            # along the first dimension (0, row) of aggregated
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            ### Perform residual connection
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features, 
        # sums the state of all nodes in each graph
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        mean, std = torch.chunk(self.encoder_net(graph_state), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) #dim: 100, M

class GaussianEncoderConvolution(nn.Module):
    def __init__(self, node_feature_dim, filter_length,M):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, 2*M)

        self.cached = False

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        ------- 
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)
 
        # ---------------------------------------------------------------------------------------------------------

        # Implementation in spectral domain
        L, U = torch.linalg.eigh(A)        
        exponentiated_L = L.unsqueeze(2).pow(torch.arange(self.filter_length, device=L.device))
        diagonal_filter = (self.h[None,None] * exponentiated_L).sum(2, keepdim=True)
        node_state = U @ (diagonal_filter * (U.transpose(1, 2) @ X))

        # ---------------------------------------------------------------------------------------------------------

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        mean, std = torch.chunk(self.output_net(graph_state), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) #dim: 100, M


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net


    def embed_graph_size(self, graph_size, target=None):
        """
        unpack bits of graph_size into a tensor of shape [batch, 5]

        Parameters:
        graph_size: [torch.Tensor] (int)
           A tensor of dimension `(batch_size)` containing the number of nodes in each graph.
        """

        if isinstance(graph_size, torch.Tensor):
            graph_size = graph_size.numpy()

        assert isinstance(graph_size, np.ndarray), 'graph_size must be a numpy array'
        assert graph_size.ndim == 1, 'graph_size must be a 1D array'

        graph_size = graph_size[:, None].astype(np.uint8)
        graph_size = np.unpackbits(graph_size, axis=1)[:,-5:] # hard coded 5 bits (max 28 nodes)
        graph_size = torch.tensor(graph_size).to(target)
        return graph_size

    def forward(self, z, graph_sizes):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        graph_sizes = self.embed_graph_size(graph_sizes, z)
        z = torch.cat([z, graph_sizes], dim=1)

        logits = self.decoder_net(z) # batch, 28,28
        probs = torch.sigmoid(logits)
        probs = torch.tril(probs, diagonal=-1)
        
        return td.Independent(td.Bernoulli(probs=probs), 2)
    

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder


    def elbo(self,  x, edge_index, batch):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim)`
        edge_index: [torch.Tensor]
           A tensor of dimension `(2, num_edges)`
        batch: [torch.Tensor]
           A tensor of dimension `(num_nodes)`
        """
        # q = self.encoder(x, edge_index, batch)


        # elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) # dim: 64
        q = self.encoder(x, edge_index, batch)  # returns: distribution of shape [batch=100, latent_dim=1]
        z = q.rsample() # dim: batch, latent_dim:  100, 1

        A = to_dense_adj(edge_index, batch) # dim: 100, 28, 28
        A = torch.tril(A, diagonal=-1) # dim: 100, 28, 28


        _, graph_sizes = batch.unique(return_counts=True)

        log_prob = self.decoder(z, graph_sizes).log_prob(A) # 100

        # kl divergence estimation
        kl = td.kl_divergence(q, self.prior())

        return (log_prob - kl).mean()

    def sample(self, n_samples=None, return_mean: bool=False, graph_sizes=None, z=None, mask=True):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        
        n_samples = n_samples or 1 # overwritten if graph_sizes is not None
        graph_sizes = SAMPLES_GRAPH_SIZE_FROM_DATA(n_samples) if graph_sizes is None else graph_sizes

        if z is None: z = self.prior().sample(torch.Size([len(graph_sizes)]))
        else: assert z.shape[0] == len(graph_sizes), 'z must have the same size as graph_sizes'
        
        posterior: td.Distribution = self.decoder(z, graph_sizes)

        if return_mean:
            sample = posterior.mean
        else:
            sample = posterior.sample()

        sample = sample + sample.transpose(1, 2)

        if mask:
            for i,N in enumerate(graph_sizes):
                sample[i, N:] = 0
                sample[i, :, N:] = 0
        return sample, graph_sizes
    
    def mean_sample(self, n_samples=1):
        """
        Sample the mean from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return torch.mean(self.decoder(z).sample(torch.Size([200])), dim=0)
    
    def forward(self,  x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo( x, edge_index, batch)


def train(model, optimizer, train_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    train_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    # Fit the model
    # Number of epochs

    train_losses = []

    model.train()
    tqdm_range = tqdm(range(epochs))
    for epoch in tqdm_range:
        # Loop over training batches
        
        train_loss = 0.
        for data in train_loader:
            loss = model(data.x, data.edge_index, batch=data.batch)

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)

        tqdm_range.set_postfix({'train_loss': train_loss})
        train_losses.append(train_loss)
    return train_losses


if __name__ == "__main__":
    # Parse arguments
    import argparse
    import glob

    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--encoder', type=str, default='mm', choices=['mm','conv','mp_node','mp_conv'], help='Prior distribution (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaus', choices=['gaus'], help='Prior distribution (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model_mp_node.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load the MUTAG dataset
    # Load data
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    node_feature_dim = 7

    dataloader = DataLoader(dataset, batch_size=188)

    # Instantiate the model
    state_dim = 8
    num_message_passing_rounds = 4

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    decoder_net = nn.Sequential(
        # nn.Linear(M, 10),
        nn.Linear(M + 5, 10),
        nn.ReLU(),
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 784),
        nn.Unflatten(-1, (28, 28))
    )

    if args.encoder == 'mp_node' or args.encoder == 'mp_conv':
        decoder = BernoulliNodeDecoder(decoder_net)
    else:
        decoder = BernoulliDecoder(decoder_net)
    
    
    # Define VAE model
    if args.encoder == 'conv':
        encoder = GaussianEncoderConvolution(node_feature_dim, 5, M)
    elif args.encoder == 'mp_node' or args.encoder == 'mp_conv':
        encoder = GaussianEncoderNodeMessagePassing(node_feature_dim, state_dim, num_message_passing_rounds, M)
    elif args.encoder == 'mp_conv':
        encoder = GaussianEncoderNodeConvolution(node_feature_dim, 5, M)
    else:
        encoder = GaussianEncoderMessagePassing(node_feature_dim, state_dim, num_message_passing_rounds, M, dropout=0.03)
    if args.encoder == 'mp_node' or args.encoder == 'mp_conv':
        model = VAENode(prior, decoder, encoder).to(device)
    else:
        model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Train model
        train(model, optimizer, dataloader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)

        print('asfdsaf',prior.mean, prior.std)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        # Generate samples
        model.eval()
        with torch.no_grad():
            samples, graph_sizes = model.sample(64)
            samples = samples.cpu()
            save_image(samples.view(-1, 1, 28, 28), args.samples)

            means, graph_sizes_means = model.sample(graph_sizes=torch.arange(17,29), mask=False, z=torch.zeros(29-17, args.latent_dim), return_mean=True)
            n = len(means)
            
            fig, axs = plt.subplots(2, n//2, figsize=(16, 6))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(means[i], cmap='binary')
                ax.set_title(f'{17+i}')
            fig.tight_layout()
            fig.savefig('assets/mean_samples.pdf', bbox_inches='tight')


            fig, axs = plt.subplots(8, np.ceil(len(samples)/8).astype(int), figsize=(16, 16))
            for i, ax in enumerate(axs.flatten()):
                G = nx.from_numpy_array(samples[i, :graph_sizes[i], :graph_sizes[i]].detach().numpy())
                nx.draw(G, ax=ax, node_size=10, node_color='black', edge_color='black')
                ax.set_title(f'size {graph_sizes[i]}')
            fig.tight_layout()
            # plt.show()
            plt.savefig('assets/samples_draw.pdf', bbox_inches='tight')
            plt.cla()
        
        
            graphs, nx_graphs = erdos_model.generate(64, return_nx_graphs=True)
            fig, axs = plt.subplots(8, np.ceil(len(graphs)/8).astype(int), figsize=(16, 16))
            for i, ax in enumerate(axs.flatten()):
                G = nx_graphs[i]
                nx.draw(G, ax=ax, node_size=10, node_color='black', edge_color='black')
                ax.set_title(f'size {G.number_of_nodes()}')
            fig.tight_layout()
            # plt.show()
            plt.savefig('assets/erdos_samples_draw.pdf', bbox_inches='tight')
            plt.cla()
        

    
    if (args.mode == 'eval') or (args.mode == 'train'):
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))


        # Generate samples
        model.eval()


        for data in dataloader:
            out = model.encoder(data.x, data.edge_index, data.batch)
        plt.scatter(*out.mean.detach().cpu().numpy().T)
        plt.show()


        gnn_adjs = model.sample(1000, return_mean=False)[0].detach().cpu().numpy().astype(bool)

        # results = list()
        # for t in tqdm(np.linspace(0, 1, 10)):
        #     A = (gnn_adjs > t)
        #     results.append(evaluate(A))
        # import matplotlib.pyplot as plt
        # results = np.array(results).T
        # for a in results:
        #     plt.plot(a)
        # plt.show()

        erdos = erdos_model.generate(1000)
        ERDOS_adjcs = np.stack([
            np.pad(arr := er.todense(), ((0, 28-len(arr)), (0, 28-len(arr)))).astype(bool)
            for er in erdos
        ])

        print('novel, unique, novel+unique')
        
        named_data = (
            ('MUTAG', MUTAG_adjcs),
            ('ERDOS', ERDOS_adjcs),
            ('GNN', gnn_adjs),
        )

        colors = ["red", "blue", "green"]

        for t, A in named_data:
            print(t)
            print(evaluate(A))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for i, fn in enumerate((plot_node_degree_histogram, plot_clustering_coefficient_histogram, plot_eigenvector_centrality)):
            fn(*named_data, ax=axs[i], colors=colors, mean_over_graph=False, alpha=0.5)
        fig.tight_layout()
        fig.savefig('assets/results_plot.pdf', bbox_inches='tight', dpi=300)
        plt.show()


