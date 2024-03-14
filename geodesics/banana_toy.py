import math

import numpy as np

import torch
import torch.distributions as td


class GaussianMixtureModel(td.MixtureSameFamily):

    def __init__(self, log_w, mean, log_std, **kwargs):
        """
        Gaussian mixture model

        Parameters:
        -----------
        * log_w: log of the weights assigned to each of the components of the mixture.
                 exp(log_w) must sum to 1
        * mean: the means of the components
        * log_std: the log standard deviations
        * **kwargs keyword args to `td.MixtureSameFamily` (the parent class of this one)
        """
        w = torch.exp(log_w)
        std = torch.exp(log_std)

        mix = td.Categorical(w)
        gauss = td.Independent(td.Normal(loc=mean, scale=std), 1)
        super().__init__(mix, gauss, **kwargs)


def get_banana_metric(sigma, epsilon, banana_path):
    """
    load banana toy dataset and return the induced metric.
    The metric is a 1/(p(x) + epsilon) where p(x) is a 
    gaussian mixture with means in the data point and
    standard deviation sigma

    Parameters:
    ---------
     * sigma: standard deviation of the components of the mixture
     * epsilon: small constant making sure we don't devide by zero
    
    Returns:
    ----------
     * metric: Callable[[torch.tensor], torch.tensor] - the metric. Returns the
      metric mentioned in the description for each point. So it takes in a 
      vector `points` of shape (N, D), and outputs a Tensor of shape (N, D, D)
      i.e. one DxD matrix for each point at which it is to be evaluated.
      Note that in this case, D = 2

     * data: torch.tensor of shape (N, 2)
    """

    # load data
    data = torch.from_numpy(np.load(banana_path)).float()
    N, d = data.shape

    #create the mixture
    mixture = GaussianMixtureModel(
        log_w=-math.log(N)*torch.ones(N),
        mean=data,
        log_std=math.log(sigma)*torch.ones(N,d))
    
    #create the metric function
    def metric(points):
        n = points.shape[0]
        log_prob = mixture.log_prob(points) # log p(x)
        m = (1/(epsilon + torch.exp(log_prob)).view(n, 1, 1)*torch.eye(d).view(1,d,d)).float()
        return m
    
    return metric, data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import get_shortest_path

    metric, data = get_banana_metric(0.15, 1e-1, 'data/toybanana.npy')

    xmin, ymin = tuple(data.min(dim=0).values.tolist())
    xmax, ymax = tuple(data.max(dim=0).values.tolist())

    pad = 0.2
    x = torch.linspace(xmin-pad, xmax+pad, 100)
    y = torch.linspace(ymin-pad, ymax+pad, 101)

    X, Y = torch.meshgrid(x, y, indexing='xy')

    G = metric(torch.stack((X.flatten(),Y.flatten()), dim=1))
    V = torch.linalg.det(G)
    
    v = torch.reshape(V, X.shape)

    plt.pcolormesh(x, y, v, cmap=plt.cm.RdBu_r, clim=(v.min(), v.max()), shading='nearest')
    plt.colorbar()
    plt.scatter(data[:, 0], data[:, 1], s=10, c='k')

    point_0 = torch.tensor([-1.2, -0.4], requires_grad=False)
    point_1 = torch.tensor([-2, 0.4], requires_grad=False)
    point_0 = torch.tensor([1.4, -0.5], requires_grad=False)

    final_curve, initial_curve = get_shortest_path(
        point_0=point_0,
        point_1=point_1,
        n=30,
        emb_dim=2,
        curve_degree=10,
        metric=metric,
        return_initial_curve=True
    )

    initial_curve = initial_curve.detach().numpy()
    plt.plot(initial_curve[:, 0], initial_curve[:, 1], '-o',  color='C1', label='initial curve')

    final_curve = final_curve.detach().numpy()
    plt.plot(final_curve[:, 0], final_curve[:, 1], '-o',  color='C2', label='optimized curve')

    plt.plot(initial_curve[[0, -1], 0], initial_curve[[0, -1], 1], 'o', color='k', label='points')

    plt.legend()

    plt.show()
