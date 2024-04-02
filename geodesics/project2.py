# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen and Søren Hauberg, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import os
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.utils.data
from curve_fitter import (
    Curve2Energy,
    CurveConfig,
    OptimizerClass,
    PiecewiseCurveFitter,
    PolynomialCurveFitter,
)
from torch.distributions.kl import kl_divergence as KL
from tqdm import tqdm

if os.path.split(os.getcwd())[-1] == 'adv_ml_project':
    os.chdir('./geodesics')
from copy import deepcopy

import einops


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

##### we did the below.

class VAEENSEMBLE(nn.Module):

    def __init__(self, prior: GaussianPrior, get_decoder: Callable[[], nn.Module], encoder: GaussianEncoder, num_models=10):
        super(VAEENSEMBLE, self).__init__()
        self.num_models = num_models
        self.prior = prior
        self.encoder = encoder
        self.decoders = nn.ModuleList([BernoulliDecoder(get_decoder()) for _ in range(num_models)])
    
    def sample_decoder(self, num_samples=1):
        return (self.decoders[i] for i in np.random.choice(self.num_models, num_samples, replace=self.num_models==1))

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        decoder, = self.sample_decoder()
        elbo = torch.mean(decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo
    
    def forward(self, x):
        return -self.elbo(x)
    
    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        decoder, = self.sample_decoder()
        return decoder(z).sample()

    # def sample_energy(self, z, num_montecarlo_samples=100) -> torch.Tensor:
    #     divergense = 0
    #     for z_i, z_j in zip(z[:-1], z[1:]):
    #         for _ in range(num_montecarlo_samples):
    #             decoder_1, decoder_2 = self.sample_decoder(2) # gets two random decoders from the list of 10
    #             divergense += KL(decoder_1(z_i), decoder_2(z_j))
    #     return divergense / num_montecarlo_samples

    def fewer_decoders(self, num_decoders=10) -> 'VAEENSEMBLE':
        this = deepcopy(self)
        this.decoders = this.decoders[:num_decoders]
        this.num_models = num_decoders
        return this
        
### we did the above

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
    pd = torch.cdist(torch.tensor(curve_points), torch.tensor(latent))  # M x N
    pd_min, _ = torch.min(pd, dim=1)
    pd_min_max = pd_min.max()
    return pd_min_max


# we did the below:
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


@torch.no_grad()
def get_entropy(model: VAE, x_resolution: int, y_resolution: int, _view: tuple[tuple[int, int], tuple[int, int]], device: str):
    xs = np.linspace(*_view[0], x_resolution)
    ys = np.linspace(*_view[1], y_resolution)
    mesh_batch = torch.tensor(np.meshgrid(xs, ys), device=device).float()
    mesh_batch = einops.rearrange(mesh_batch, 'd x y -> (x y) d')
    entropy: torch.Tensor = model.decoder(mesh_batch).entropy() #shape = (len(xs)*len(ys))
    return einops.rearrange(entropy, '(x y) -> x y', x=x_resolution)

# end of we did this...


def main():
    # Parse arguments
    import argparse
    import glob
    from pathlib import Path

    from torchvision import datasets, transforms
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='part-b', choices=['train', 'plot', 'part-a', 'train-ensemble', 'part-b'], help='what to do when running the script (default: %(default)s)')
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



    ensemble_model_path = Path(args.model) # we did this
    ensemble_model_path = ensemble_model_path.with_stem(ensemble_model_path.stem + '_ensemble') # we did this

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

    # Define VAE model # TODO: move this inside one of the if statements? probably the one just below (if args.mode == 'train':)
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

    ###### we did the rest of this file.
    
    elif args.mode == 'train-ensemble':
        num_ensemble = 10
        
        model = VAEENSEMBLE(prior, new_decoder, encoder, num_models=num_ensemble).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs * num_ensemble, args.device)
        torch.save(model.state_dict(), ensemble_model_path)

    elif args.mode in ('plot', 'part-a', 'part-b'):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        scatter_opacity = 0.5
        num_curves = 50
        num_montecarlo_samples = 1 # for ensemble # TODO: what the hell is this?????
        verbose_energies = False

        curve_fitter_class = PolynomialCurveFitter
        curve_fitter_class = PiecewiseCurveFitter # TODO: Make this an argument

        curve_config = CurveConfig(
            num_points=10,
            emb_dim=2,
            curve_degree=3,
            optimizer_kwargs=dict(lr=1, max_iter=2*100000, line_search_fn='strong_wolfe')
            # optimizer_kwargs=dict(epochs=100, lr=1e-1)
        )

        ## Load trained model
        if args.mode == 'part-b':
            model = VAEENSEMBLE(prior, new_decoder, encoder, num_models=10).to(device)
            model.load_state_dict(torch.load(ensemble_model_path, map_location=torch.device(args.device)))
        else:
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        latents, labels = get_latents(model, mnist_train_loader, args.device)
        latents_np, labels_np = latents.detach().cpu().numpy(), labels.detach().cpu().numpy()

        ## Plot training data
        colors = np.array([f'C{i}' for i in range(num_classes)])[labels_np]
        markerfacecolor = lambda idx: list(mcolors.TABLEAU_COLORS.values())[idx] + f'{hex(int(255*scatter_opacity)):<04}'[2:]
        legend_handles = [Line2D([0], [0], label=i, marker='o', linestyle='', markeredgecolor=f'C{i}', markerfacecolor=markerfacecolor(i)) for i in range(num_classes)]

        if args.mode == 'plot':
            plt.figure()
            plt.title('Latent space')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.scatter(latents_np[:, 0], latents_np[:, 1], c=colors, alpha=scatter_opacity, s=5)
            plt.legend(handles=legend_handles)
            plt.savefig(os.path.join(args.plot_dir, 'latent_space.pdf'))
            plt.show()
            return

        elif args.mode == 'part-a':
            # Plot random geodesics, with same seed
            with torch.random.fork_rng():
                torch.random.manual_seed(4269)  # Specify the seed value
                curve_indices = torch.randint(num_train_data, (num_curves, 2))  # (num_curves) x 2 # TODO: maybe rewrite with `choice` to avoid curves starting and stopping in the same point.

            #curve_config.decoder = lambda z: model.decoder(z).mean.view(-1, 28**2) # energy from euclidian distance # TODO: replace mean with sample? - we are gonna compute KL divergences later?? so maybe neither make sense?
            curve_config.decoder = lambda z: [model.decoder(z_i) for z_i in z] # energy from Fisher-Rao # TODO: Make into argument # TODO: why not just make one forward pass?
            curve_fitter = curve_fitter_class(curve_config, device=device, verbose_energies=verbose_energies)

            x_resolution = y_resolution = 100*2
            _dist = 8
            _view = ((-_dist, _dist), (-_dist, _dist))
            entropy = get_entropy(model, x_resolution, y_resolution, _view, device) # TODO: why do we use entropy??

            # plots latent variables
            pos = plt.matshow(entropy.cpu().numpy(), extent=[*_view[0], *_view[1]], cmap='viridis', origin='lower')
            plt.scatter(latents_np[:, 0], latents_np[:, 1], c=colors, alpha=scatter_opacity, s=10)
            plt.colorbar(pos)

            geodesics_energy = list()
            for k in tqdm(range(num_curves), "Generating geodesics"):
                i = curve_indices[k, 0]
                j = curve_indices[k, 1]
                z0 = latents[i]
                z1 = latents[j]
                # TODO: Compute, and plot geodesic between z0 and z1
                curve_fitter.reset(z0, z1)
                curve_fitter.fit()
                curve = curve_fitter.points.detach().cpu().numpy()
                geodesics_energy.append(curve_fitter.forward().detach().cpu().numpy())
                
                z0, z1 = z0.detach().cpu().numpy(), z1.detach().cpu().numpy()
                plt.plot(curve[:, 0], curve[:, 1], c='k')
                plt.plot([z0[0], z1[0]], [z0[1], z1[1]], 'o', c='k', markersize=3)
            
            print('mean energy', np.mean(geodesics_energy))
            legend_handles.extend((Line2D([0], [0], label='geodesic', linestyle='-', color='k'),
                                Line2D([0], [0], label='endpoints', linestyle='', marker='o', markeredgecolor='k', markerfacecolor='k')))

            plt.legend(handles=legend_handles)
            plt.title('Latent space')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.savefig(os.path.join(args.plot_dir, 'latent_space_part_a.pdf'))
            plt.show()
            return
        
        elif args.mode == 'part-b':
            with torch.random.fork_rng():
                torch.random.manual_seed(4269)  # Specify the seed value
                curve_indices = torch.randint(num_train_data, (num_curves, 2)) # (num_curves) x 2
              
            curve_config.optimizer_kwargs=dict(epochs=200, lr=1e-1)
            # curve_config.optimizer_kwargs = dict(lr=1*0.2, max_iter=5*100000, line_search_fn='strong_wolfe')
            curve_config.curve_to_energy = Curve2Energy.KL #! Need PASS because we are using the ensemble model
            curve_config.optimizer_class = OptimizerClass.ADAM # ? Try m
            # curve_config.decoder = lambda z: model.sample_energy(z, num_montecarlo_samples)
            # curve_config.decoder = lambda z: model.decode(z)
            curve_config.decoder = lambda z: [
                [dec(z_i) for dec in model.sample_decoder(num_montecarlo_samples)] for z_i in z
            ]

            curve_fitter = curve_fitter_class(curve_config, device=device, verbose_energies=verbose_energies)

            x_resolution = y_resolution = 100*2
            _dist = 8
            _view = ((-_dist, _dist), (-_dist, _dist))

            entropies = list()
            for _decoder in model.decoders:
                _model = VAE(model.prior, _decoder, model.encoder).to(device).eval()
                entropies.append(get_entropy(_model, x_resolution, y_resolution, _view, device))
            entropy = torch.stack(entropies).mean(dim=0)
            pos = plt.matshow(entropy.cpu().numpy(), extent=[*_view[0], *_view[1]], cmap='viridis', origin='lower')
            plt.scatter(latents_np[:, 0], latents_np[:, 1], c=colors, alpha=scatter_opacity, s=10)
            plt.colorbar(pos)

            curves_10_decoders = list()
            for k in tqdm(range(num_curves), "Generating geodesics"):
                i = curve_indices[k, 0]
                j = curve_indices[k, 1]
                z0 = latents[i]
                z1 = latents[j]
                curve_fitter.reset(z0, z1)
                curve_fitter.fit()
                curve = curve_fitter.points.detach().cpu().numpy()
                curves_10_decoders.append(curve)

                z0, z1 = z0.detach().cpu().numpy(), z1.detach().cpu().numpy()
                plt.plot(curve[:, 0], curve[:, 1], c='k')
                plt.plot([z0[0], z1[0]], [z0[1], z1[1]], 'o', c='k', markersize=3)
            
            legend_handles.extend((Line2D([0], [0], label='geodesic', linestyle='-', color='k'),
                                Line2D([0], [0], label='endpoints', linestyle='', marker='o', markeredgecolor='k', markerfacecolor='k')))

            plt.title('Latent space')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.legend(handles=legend_handles)
            plt.savefig(os.path.join(args.plot_dir, 'latent_space_part_b.pdf'))
            plt.show()

            list_of_lists_of_curves = list()
            num_models_list = list(range(1,9+1))
            for num_decoders in num_models_list:
                _model = model.fewer_decoders(num_decoders)
                curve_fitter.decoder = lambda z: _model.sample_energy(z, num_montecarlo_samples)
                _curves = list()
                for k in tqdm(range(num_curves), "Generating geodesics"):
                    i = curve_indices[k, 0]
                    j = curve_indices[k, 1]
                    z0 = latents[i]
                    z1 = latents[j]
                    curve_fitter.reset(z0, z1)
                    curve_fitter.fit()
                    curve = curve_fitter.points.detach().cpu().numpy()
                    _curves.append(curve)

                    z0, z1 = z0.detach().cpu().numpy(), z1.detach().cpu().numpy()
                    plt.plot(curve[:, 0], curve[:, 1], c='k')
                    plt.plot([z0[0], z1[0]], [z0[1], z1[1]], 'o', c='k')
                list_of_lists_of_curves.append(_curves)
            list_of_lists_of_curves.append(curves_10_decoders)

            _func = partial(proximity, latent=latents)
            mean_proximity = [np.mean([_func(curve) for curve in curves]) for curves in list_of_lists_of_curves]
            plt.figure()
            plt.plot(num_models_list + [10], mean_proximity, label='mean proximity')
            plt.xlabel('number of decoders')
            plt.ylabel('mean proximity')
            plt.legend()
            plt.savefig(os.path.join(args.plot_dir, 'mean_proximity.pdf'))
            plt.show()

            return


if __name__ == '__main__':
    main()
