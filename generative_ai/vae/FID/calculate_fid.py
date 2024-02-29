from typing import Callable

import tqdm
import numpy as np
from scipy import linalg

import torch

from FID.model import MiniModel


real_encodings = np.load('assets/test_encodings_for_fid.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniModel().to(device)
model.load_state_dict(torch.load('assets/mini_classifier.pth'))
model.eval()


def frechet_distance(mu1, sigma1, mu2, sigma2):
    # https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
    return sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * linalg.sqrtm(sigma1@sigma2))


def calculate_fid(get_encoding: Callable[[], torch.Tensor], num_samples: int=10_000, verbose: bool=True) -> float:
    """
    Calculate the Frechet Inception Distance (FID) between the real encodings and the generated encodings.

    Args:
        get_encoding (Callable[[], torch.Tensor]): A function that returns a batch of encodings whenever called e.g. `lambda: model.sample(n=32)`.
        num_samples (int, optional): Number of samples to generate. Defaults to 10_000.
        verbose (bool, optional): Whether to display a progress bar. Defaults to True.
    """

    fake_encodings = np.empty((num_samples, real_encodings.shape[1]))

    with torch.no_grad():
        num_sampled = 0
        pbar = tqdm.tqdm(total=num_samples, desc='FID', disable=not verbose)
        while num_sampled < num_samples:
            sample = get_encoding().to(device)
            batch_size = min(sample.shape[0], num_samples - num_sampled)
            encoding = model.encode(sample)
            fake_encodings[num_sampled:num_sampled+batch_size] = encoding.cpu().numpy()[:batch_size]
            num_sampled += batch_size
            pbar.update(batch_size)
        pbar.close()

    mu1, sigma1 = real_encodings.mean(axis=0), np.cov(real_encodings, rowvar=False)
    mu2, sigma2 = fake_encodings.mean(axis=0), np.cov(fake_encodings, rowvar=False)

    return frechet_distance(mu1, sigma1, mu2, sigma2)
