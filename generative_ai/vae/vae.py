# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import pdb

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.utils.data
from flow import Flow, GaussianBase, MaskedCouplingLayer
from scipy import stats
from scipy.stats import kde
from sklearn.decomposition import PCA
from torch.nn import functional as F
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
    
class VampPrior(nn.Module):
    def __init__(self, K, encoder):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
            Number of mixture of the approximate posterior
        """
        super(VampPrior, self).__init__()
        self.M = M
        self.K = K
        self.encoder = encoder
        self.mean = nn.Parameter(torch.randn(self.K, 28,28), requires_grad=True)
        self.w = nn.Parameter(torch.ones(self.K)/self.K, requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        #q_dist = self.encoder(self.mean) # dim: self.K, latent_dim
        
        #return q_dist # dim: latent_dim
    
        comp = self.encoder(self.mean)
        mix = td.Categorical(probs=self.w)
        gmm = td.MixtureSameFamily(mix, comp)
        return gmm

class MixtureGaussianPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussian prior distribution with zero mean and unit variance.

        Parameters:
            M: [int] 
                Dimension of the latent space.
            K: [int]
                Number of Gaussian components in the mixture.
        """
        super(MixtureGaussianPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.randn(self.K, self.M), requires_grad=True)
        self.std = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.w = nn.Parameter(torch.ones(self.K)/self.K, requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        comp = td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
        gmm = td.MixtureSameFamily(td.Categorical(probs=F.sigmoid(self.w)), comp)
        return gmm
    
class FlowPrior(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a Mixture of Gaussian prior distribution with zero mean and unit variance.

        Parameters:
            base: [torch.nn.Module] 
                The base distribution for the flow.
            transformations: [list]
                List of transformations for the flow.
        """
        super(FlowPrior, self).__init__()
        self.flow = Flow(base,transformations)
    
    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return self.flow

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


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
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)
    
class MultiGaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a MultiGaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(MultiGaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        #self.std = nn.Parameter(torch.ones(28, 28)*0.2, requires_grad=True)
        self.std = nn.Parameter(torch.ones(28, 28)*0.2, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        #return td.Independent(td.MultivariateNormal(logits, torch.diag_embed(torch.exp(self.std))), 2)
        return td.Independent(td.Normal(logits, self.std), 2)



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

    def iwae_elbo(self, x, n_samples=5):
        """
        Compute the IWAE ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
        A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        n_samples: [int]
        Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)  # returns: distribution
        z = q.rsample(torch.Size([n_samples]))  # dim: n_samples, batch, latent_dim

        log_prob = self.decoder(z).log_prob(x.unsqueeze(0))  # in 32, 28, 28 out dim: n_samples, batch
        log_qz = q.log_prob(z)  # dim: n_samples, batch

        if isinstance(self.prior, MixtureGaussianPrior) or isinstance(self.prior, FlowPrior) or isinstance(self.prior, VampPrior):
            prior_log_prob = self.prior().log_prob(z.view(-1,z.shape[2])) # dim: (n_samples*batch), latent_dim
            prior_log_prob = prior_log_prob.view(n_samples, -1) # dim: n_samples, batch
        else:
            prior_log_prob = self.prior().log_prob(z)  # dim: n_samples, batch

        weights = log_prob + prior_log_prob - log_qz  # dim: n_samples, batch
        weights = weights.transpose(0, 1)  # dim: batch, n_samples

        # Compute log mean exp to avoid numerical instability.
        max_weights = weights.max(dim=1, keepdim=True)[0]  # dim: batch, 1
        weights = weights - max_weights  # dim: batch, n_samples
        weights = torch.log(torch.mean(torch.exp(weights), dim=1))  # dim: batch
        weights = weights + max_weights.squeeze()  # dim: batch

        return torch.mean(weights)


    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)  # returns: distribution
        z = q.rsample() # dim: batch, latent_dim:  64, 32
        log_prob = self.decoder(z).log_prob(x) # in 32, 28, 28 out dim: batch

        # kl divergence estimation
        if isinstance(self.prior, MixtureGaussianPrior) or isinstance(self.prior, FlowPrior) or isinstance(self.prior, VampPrior):
            # use monte carlo estimation
            k = 256
            z_samples = q.rsample(torch.Size([k]))  # dim: k, batch, latent_dim
            z_samples_log_prob = q.log_prob(z_samples) # dim: k, batch
            prior_log_prob = self.prior().log_prob(z_samples.view(-1,z_samples.shape[2])) # dim: (k*batch), latent_dim
            prior_log_prob = prior_log_prob.view(k, -1) # dim: k, batch
            
            kl = torch.mean(z_samples_log_prob - prior_log_prob, axis=0)  # dim: batch
        else:
            kl = td.kl_divergence(q, self.prior()) # dim: batch
        
        return torch.mean(log_prob - kl, dim=0) # dim: number

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
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.iwae_elbo(x) #-self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    i = 0
    rolling_average = 30
    average = np.zeros(rolling_average)
    
    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            average[i % rolling_average] =loss.item()
            progress_bar.set_postfix(loss=np.mean(average), epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
            i += 1

def plot_data(ax, X, y, alpha=0.8, title=None):
    ### CREDIT: FROM BAYESIAN MACHINE LEARNING 02477 ###
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple']
    labels = ['digit=0', 'digit=1', 'digit=2', 'digit=3', 'digit=4', 'digit=5', 'digit=6', 'digit=7', 'digit=8', 'digit=9']

    for i in range(10):
        ax.plot(X[y==i, 0], X[y==i, 1], colors[i], label=labels[i], linestyle='None', marker='o')

    ax.set(xlabel='Axis 1', ylabel='Axis 2')
    ax.legend()

    if title:
        ax.set_title(title, fontweight='bold')
        
        
def plot_distribution(ax, density_fun, color=None, visibility=1, label=None, title=None, num_points = 100):
    ### CREDIT: FROM BAYESIAN MACHINE LEARNING 02477 ###
    
    # create grid for parameters (a,b)
    a_array = np.linspace(-5, 5, num_points)
    b_array = np.linspace(-5, 5, num_points)
    A_array, B_array = np.meshgrid(a_array, b_array)   
    
    # form array with all combinations of (a,b) in our grid
    AB = torch.tensor(np.column_stack((A_array.ravel(), B_array.ravel())),dtype=torch.float32)
    
    # evaluate density for every point in the grid and reshape bac
    Z = density_fun(AB).detach().cpu().numpy()
    Z = Z.reshape((len(a_array), len(b_array)))
    
    # plot contour  
    ax.contour(a_array, b_array, Z, colors=color, alpha=visibility)
    ax.plot([-1000], [-1000], color=color, label=label)
    ax.set(xlabel='axis1', ylabel='axis2', xlim=(-5, 5), ylim=(-5, 5), title=title)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    import glob

    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'vis', 'msample', 'fid'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaus', choices=['gaus', 'mog', 'flow','vampprior'], help='Prior distribution (default: %(default)s)')
    parser.add_argument('--model', type=str, default='vae/model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='vae/samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--notBinarize', action='store_true', help='binarize the data (default: True)')

    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    M = args.latent_dim

    if args.notBinarize:
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float().squeeze())])
    else:
        threshold = 0.5
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])
        
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True, transform=data_transform), batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,transform=data_transform), batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    if args.notBinarize:
        decoder = MultiGaussianDecoder(decoder_net)
    else:
        decoder = BernoulliDecoder(decoder_net)
    
    # Define VAE model
    encoder = GaussianEncoder(encoder_net)
    
    # Define prior distribution
    if args.prior == 'gaus':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MixtureGaussianPrior(M, 5)
    elif args.prior == 'vampprior':
        prior = VampPrior(15, encoder)
    elif args.prior == 'flow':
        base = GaussianBase(M)
        # Define transformations
        
        num_transformations = 5*4
        num_hidden = 8*2

        # Make a mask that is 1 for the first half of the features and 0 for the second half
        mask = torch.zeros((M,))
        mask[M//2:] = 1

        transformations =[]
        for i in range(num_transformations):
            mask = (1-mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        
        prior = FlowPrior(base,transformations)

    
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    elif args.mode == 'msample':
        # samples the mean of the decoder
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.mean_sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        loss_list = []
        with torch.no_grad():
            data_iter = iter(mnist_test_loader)
            for x in tqdm(data_iter):
                x = x[0].to(args.device)
                loss = model(x)
                loss_list.append(loss.item())
        

        print(f'ELBO: {sum(loss_list)/len(loss_list)}')
        
    elif args.mode == 'vis':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        loss_list = []
        with torch.no_grad():
            data_iter = iter(mnist_test_loader)
            
            # gets samples from multiple batches
            n_batches = 20
            z = np.zeros((args.batch_size*n_batches, M))
            y = np.zeros(args.batch_size*n_batches)
            for i in range(n_batches):
                x, y_sample = next(data_iter)
                x = x.to(args.device)
                q = model.encoder(x)
                z_sample = q.rsample()  # batch, dim: 32, 32
                z[i*args.batch_size:(i+1)*args.batch_size] = z_sample.cpu().numpy()
                y[i*args.batch_size:(i+1)*args.batch_size] = y_sample.numpy()
            
        if M != 2:
            assert False, "Only 2D latent space is supported for visualization" 
        
        fig, ax = plt.subplots()
        
        
        fig, axes = plt.subplots(1, figsize=(8,8))
        plot_data(axes, z,y, alpha=0.5)
        plot_distribution(axes, density_fun=model.prior().log_prob, color='b', title=f'MINST samples from approximate posterior \n With ln {args.prior} prior distribution',num_points=1000,visibility=0.7)

        fig.savefig(args.samples)    
        
    elif args.mode == 'fid':
        from FID.calculate_fid import calculate_fid

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        fid = calculate_fid(lambda: torch.reshape((x:=model.sample(64)), (*x.shape[:-2], -1)), num_samples=10000, verbose=True)

        print(f'FID: {fid:.4f}')