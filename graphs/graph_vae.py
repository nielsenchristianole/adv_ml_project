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

class GaussianEncoder(nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, M):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim # 7
        self.state_dim = state_dim # 16
        self.num_message_passing_rounds = num_message_passing_rounds # 4

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.Dropout(0.1),
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
        state = self.input_net(x)
        #state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

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

    # def forward(self, x):
    #     """
    #     Given a batch of data, return a Gaussian distribution over the latent space.

    #     Parameters:
    #     x: [torch.Tensor] 
    #        A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
    #     """
    #     mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
    #     return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z, batch):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        unique, count = torch.unique(batch, return_counts=True)

        # make logits mask for each graph based on unique and count. where unique is which graph and count is how many nodes in each graph
        mask = torch.zeros(len(unique), 28, 28)
        for u in unique:
            mask[u, :count[u], :count[u]] = 1

        logits = logits * mask
        # Set upper triangular part of the logits to 0
        with torch.no_grad():
            logits = torch.tril(logits, diagonal=-1)    
        
        return td.Independent(td.Bernoulli(logits=logits), 2)
    

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
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        # q = self.encoder(x, edge_index, batch)

        # elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) # dim: 64
        q = self.encoder(x, edge_index, batch)  # returns: distribution of shape [batch=100, latent_dim=1]
        z = q.rsample() # dim: batch, latent_dim:  100, 1

        A = to_dense_adj(edge_index, batch) # dim: 100, 28, 28

        # permute a 100 times but only inside the mask
        # create 0 matrix of shape 5, 100,28,28
        A_perm = torch.zeros(5, len(A), 28, 28)
        unique, count = torch.unique(batch, return_counts=True)
        for i in range(len(A)):
            for j in range(len(A_perm)):
                perm = torch.randperm(count[i])
                A_perm[j,i,:count[i], :count[i]] = A[i, perm,:][:,perm]

        

        A_perm = torch.tril(A_perm, diagonal=-1) # dim: 5, 100, 28, 28
        x_star = self.decoder(z, batch)

        log_prob = self.decoder(z, batch).log_prob(A_perm) # 5, 100
        log_prob = torch.max(log_prob,axis=0)[0]

    
        # kl divergence estimation
        kl = td.kl_divergence(q, self.prior()).sum(-1)

        return torch.mean(log_prob - kl, dim=0)

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
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


def train(model, criterion, optimizer, scheduler, train_loader, epochs, device):
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

    validation_losses = []


    model.train()
    tqdm_range = tqdm(range(epochs))
    for epoch in tqdm_range:
        # Loop over training batches
        
        train_loss = 0.
        for data in train_loader:
            out = model(data.x, data.edge_index, batch=data.batch)
            loss = out#cross_entropy(out, data.y.float())

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)

        
        scheduler.step()
        tqdm_range.set_postfix({'train_loss': train_loss})


        # Validation, print and plots
        with torch.no_grad():    
            model.eval()
            # Compute validation loss and accuracy
            validation_loss = 0.
            validation_accuracy = 0.
            for data in validation_loader:
                out = model(data.x, data.edge_index, data.batch)
                validation_accuracy += sum((out>0) == data.y).cpu() / len(validation_loader.dataset)
                validation_loss += out#cross_entropy(out, data.y.float()).cpu().item() * data.batch_size / len(validation_loader.dataset)

            # Store the training and validation accuracy and loss for plotting

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

        
        model.train()

if __name__ == "__main__":
    # Parse arguments
    import argparse
    import glob

    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaus', choices=['gaus'], help='Prior distribution (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=5, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load the MUTAG dataset
    # Load data
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    node_feature_dim = 7

    # Split into training and validation
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

    # Create dataloader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=100)
    validation_loader = DataLoader(validation_dataset, batch_size=44)
    test_loader = DataLoader(test_dataset, batch_size=44)

    # Instantiate the model
    state_dim = 8
    num_message_passing_rounds = 4

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    decoder_net = nn.Sequential(
        nn.Linear(M, 10),
        nn.ReLU(),
        nn.Linear(10, 784),
        nn.Unflatten(-1, (28, 28))
    )

    decoder = BernoulliDecoder(decoder_net)
    
    # Define VAE model
    encoder = GaussianEncoder(node_feature_dim, state_dim, num_message_passing_rounds, M)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)#0.995)

        # Train model
        train(model, criterion, optimizer, scheduler, train_loader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        pass