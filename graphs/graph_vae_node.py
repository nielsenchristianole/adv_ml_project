import pdb

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.utils.data
from scipy import stats
from scipy.stats import kde
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
from tqdm import tqdm


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
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoderNodeMessagePassing(nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, M):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoderNodeMessagePassing, self).__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim # 7
        self.state_dim = state_dim # 16
        self.num_message_passing_rounds = num_message_passing_rounds # 4

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            #torch.nn.Dropout(0.1),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                #torch.nn.Dropout(0.1),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                #torch.nn.Dropout(0.1),
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
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            ### Perform residual connection
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features
        means, stds = torch.chunk(self.encoder_net(state), 2, dim=-1)

        return td.Independent(td.Normal(loc=means, scale=torch.exp(stds)), 1)
    
class BernoulliNodeDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliNodeDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, rx, tx):
        """
        Returns the probability of a links between nodes in lists rx and tx
        based on: return torch.sigmoid((self.embedding.weight[rx]*self.embedding.weight[tx]).sum(1) + self.bias)
        """
        #logits = self.decoder_net(z)
        
        logits= torch.sigmoid((rx*tx).sum(1) + self.bias)
        
        return td.Independent(td.Bernoulli(logits=logits), 1) # 2
    
class VAENode(nn.Module):
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
            
        super(VAENode, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self,  x, edge_index, batch):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] number of nodes
        edge_index: [torch.Tensor] edge index between nodes 
        batch: [torch.Tensor] batch index of nodes for x
        """
        # elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) # dim: 64
        
        qs = self.encoder(x, edge_index, batch)  # returns: distribution of all latent variables for all nodes
        z = qs.rsample() # sample from the distribution
        nodes_numbers = torch.arange(len(x))

        # Compute the ELBO for each graph.
        log_probs = torch.zeros(len(torch.unique(batch)))
        kls = torch.zeros(len(torch.unique(batch)))
        kls_raw = td.kl_divergence(qs, self.prior())
        for b in torch.unique(batch):  
            x_graph = x[batch == b]
            #qs_graph = qs[batch == b] ! not possible for list
            # get indices where batch == b is true
            qs_idx = torch.where(batch == b)[0]

            # get all nodes in batch b
            b_nodes = nodes_numbers[batch==b] # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])

            # get all edge_index in batch b
            b_edge_index = edge_index[:, torch.logical_or(torch.isin(edge_index[0], b_nodes), torch.isin(edge_index[1], b_nodes))]

            # Calculate elbo for each node in b
            
            # Compute the log probability between all pairs of nodes
            pairs = torch.combinations(torch.arange(x_graph.size(0)))
            targets = torch.zeros(len(pairs))
            # set target to 1 if edge exists
            for rx, tx in b_edge_index.T:
                targets[(pairs[:, 0] == rx) & (pairs[:, 1] == tx)] = 1
                
            z_b = z[qs_idx]

            link_probabilitys = self.decoder(z_b[pairs[:, 0]], z_b[pairs[:, 1]])
            log_prob_list = link_probabilitys.log_prob(targets)
            
            log_probs[b] = log_prob_list
            kls[b] = torch.sum(kls_raw[qs_idx])

        return torch.mean(log_probs - kls, dim=0)

    def sample(self, n_samples=1, *, sizes: torch.IntTensor=None, return_mean: bool=False):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        dist: td.Distribution = self.decoder(z, batch_size=n_samples, sizes=sizes)

        if return_mean:
            return dist.mean

        # import matplotlib.pyplot as plt
        # plt.ecdf(dist.mean.flatten().tolist())
        # plt.scatter(z[:,0], z[:,1])

        sample = dist.sample()
        return sample
    
    def forward(self,  x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo( x, edge_index, batch)
