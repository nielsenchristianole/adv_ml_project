import numpy as np

import torch
import torch.utils.data
from torchvision import datasets, transforms

from FID.calculate_fid import calculate_fid

BATCH_SIZE = 128
NUM_SAMPLES = 10_000

_transforms = transforms.Compose ([
    transforms.ToTensor (),
    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
    transforms.Lambda(lambda x: x.flatten())
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=_transforms),
    batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(42))

all_data = torch.cat([x for x, _ in train_loader], dim=0).numpy()


sampled_encodings_uniform = np.random.uniform(0, 1, size=(NUM_SAMPLES, 28**2))
sampled_encodings_normal = np.random.normal(all_data.mean(), all_data.std(), size=(NUM_SAMPLES, 28**2))
sampled_encodings_multivariate_normal = np.random.multivariate_normal(all_data.mean(axis=0), np.cov(all_data, rowvar=False), size=NUM_SAMPLES)


def get_next_batch(data, batch_size):
    i = 0
    while i < len(data):
        yield torch.tensor(data[i:i+batch_size], dtype=torch.float32)
        i += batch_size


generator_uniform = get_next_batch(sampled_encodings_uniform, BATCH_SIZE)
generator_normal = get_next_batch(sampled_encodings_normal, BATCH_SIZE)
generator_multivariate_normal = get_next_batch(sampled_encodings_multivariate_normal, BATCH_SIZE)

fid_training = calculate_fid(lambda: next(iter(train_loader))[0], num_samples=len(train_loader), verbose=False)
fid_uniform = calculate_fid(lambda: next(generator_uniform), num_samples=NUM_SAMPLES, verbose=False)
fid_normal = calculate_fid(lambda: next(generator_normal), num_samples=NUM_SAMPLES, verbose=False)
fid_multivariate_normal = calculate_fid(lambda: next(generator_multivariate_normal), num_samples=NUM_SAMPLES, verbose=False)

print(f'FID lower bound: {fid_training:.4f}')
print(f'FID uniform: {fid_uniform:.4f}')
print(f'FID normal: {fid_normal:.4f}')
print(f'FID multivariate normal: {fid_multivariate_normal:.4f}')
