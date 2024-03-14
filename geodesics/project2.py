# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen and Søren Hauberg, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

from typing import Callable
import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import torch.utils.data
from tqdm import tqdm
import numpy as np
import os
if os.path.split(os.getcwd())[-1] == 'adv_ml_project':
    os.chdir('./geodesics')

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
        return td.Independent(td.ContinuousBernoulli(logits=logits), 3)


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

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


class VAEENSEMBLE(nn.Module):

    def __init__(self, prior: GaussianPrior, get_decoder: Callable[[], nn.Module], encoder: GaussianEncoder, num_models=10):
        super(VAEENSEMBLE, self).__init__()
        self.num_models = num_models
        self.prior = prior
        self.encoder = encoder
        self.decoders = nn.ModuleList([BernoulliDecoder(get_decoder()) for _ in range(num_models)])
    
    def sample_decoder(self):
        return self.decoders[np.random.randint(self.num_models)]
    
    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        decoder = self.sample_decoder()
        elbo = torch.mean(decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo
    
    def forward(self, x):
        return -self.elbo(x)
    
    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        decoder = self.sample_decoder()
        return decoder(z).sample()


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
    num_steps = len(data_loader)*epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = noise(x.to(device))
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1


def proximity(curve_points, latent):
    """
    Compute the average distance between points on a curve and a collection
    of latent variables.

    Parameters:
    curve_points: [torch.tensor]
        M points along a curve in latent space. tensor shape: M x latent_dim
    latent: [torch.tensor]
        N points in latent space (latent means). tensor shape: N x latent_dim

    The function returns a scalar.
    """
    pd = torch.cdist(curve_points, latent)  # M x N
    pd_min, _ = torch.min(pd, dim=0)
    pd_min_max = pd_min.max()
    return pd_min_max

def get_latents(model: VAE, mnist_train_loader: torch.utils.data.DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ## Encode test and train data
    latents, labels = [], []
    with torch.no_grad():
        for x, y in mnist_train_loader:
            z = model.encoder(x.to(device))
            latents.append(z.mean)
            labels.append(y.to(device))
        latents = torch.concatenate(latents, dim=0)
        labels = torch.concatenate(labels, dim=0)
    return latents, labels


def main():
    from torchvision import datasets, transforms
    import glob
    from pathlib import Path

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='plot', choices=['train', 'plot', 'part-a', 'train-ensemble'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='../assets/model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--plot-dir', type=str, default='../assets/', help='file to save latent plot in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--data-dir', type=str, default='../data', help='where to store and look for data (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)
    
    num_train_data = 2048
    num_test_data = 16  # we keep this number low to only compute a few geodesics
    num_classes = 3
    train_tensors = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Define VAE model
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(new_decoder())
    model = VAE(prior, decoder, encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)
    
    elif args.mode == 'train-ensemble':
        num_ensemble = 10

        model_path = Path(args.model)
        model_path = model_path.with_stem(model_path.stem + '_ensemble')
        
        model = VAEENSEMBLE(prior, new_decoder, encoder, num_models=num_ensemble).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs * num_ensemble, args.device)
        torch.save(model.state_dict(), model_path)

    elif args.mode in ('plot', 'part-a'):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import matplotlib.colors as mcolors
        from utils import get_shortest_path

        scatter_opacity = 0.2
        curve_degree = 10
        num_curve_points = 50
        num_curves = 50

        ## Load trained model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        latents, labels = get_latents(model, mnist_train_loader, args.device)
        latents_np, labels_np = latents.detach().cpu().numpy(), labels.detach().cpu().numpy()

        ## Plot training data
        plt.figure()
        plt.title('Latent space')
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        colors = np.array([f'C{i}' for i in range(num_classes)])[labels_np]
        plt.scatter(latents_np[:, 0], latents_np[:, 1], c=colors, alpha=scatter_opacity)
        
        markerfacecolor = lambda idx: list(mcolors.TABLEAU_COLORS.values())[idx] + f'{hex(int(255*scatter_opacity)):<04}'[2:]
        legend_handles = [Line2D([0], [0], label=i, marker='o', linestyle='', markeredgecolor=f'C{i}', markerfacecolor=markerfacecolor(i)) for i in range(num_classes)]

        if args.mode == 'plot':
            plt.legend(handles=legend_handles)
            plt.savefig(os.path.join(args.plot_dir, 'latent_space.pdf'))
            plt.show()
            return

        elif args.mode == 'part-a':
            # Plot random geodesics
            curve_indices = torch.randint(num_train_data, (num_curves, 2))  # (num_curves) x 2
            for k in tqdm(range(num_curves), "Generating geodesics"):
                i = curve_indices[k, 0]
                j = curve_indices[k, 1]
                z0 = latents[i]
                z1 = latents[j]
                # TODO: Compute, and plot geodesic between z0 and z1
                decoder = lambda z: model.decoder(z).mean.view(z.shape[0], 784)
                curve = get_shortest_path(z0, z1, num_curve_points, emb_dim=args.latent_dim, curve_degree=curve_degree, decoder=decoder).detach().cpu().numpy()
                z0, z1 = z0.detach().cpu().numpy(), z1.detach().cpu().numpy()
                plt.plot(curve[:, 0], curve[:, 1], c='k')
                plt.plot([z0[0], z1[0]], [z0[1], z1[1]], 'o', c='k')
            
            legend_handles.extend((Line2D([0], [0], label='geodesic', linestyle='-', color='k'),
                                Line2D([0], [0], label='endpoints', linestyle='', marker='o', markeredgecolor='k', markerfacecolor='k')))

            plt.legend(handles=legend_handles)
            plt.savefig(os.path.join(args.plot_dir, 'latent_space_part_a.pdf'))
            plt.show()
            return


if __name__ == '__main__':
    main()