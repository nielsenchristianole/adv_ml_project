import numpy as np

import torch
import torch.utils.data
from torchvision import datasets, transforms

from FID.model import MiniModel
from FID.calculate_fid import frechet_distance, calculate_fid

BATCH_SIZE = 128

_transforms = transforms.Compose ([
    transforms.ToTensor (),
    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
    transforms.Lambda(lambda x: x.flatten())
])
test_1, test_2 = torch.utils.data.random_split(
    datasets.MNIST('data', train=False, download=True, transform=_transforms),
    (0.5, 0.5),
    generator=torch.Generator().manual_seed(42)
)
test_loader_1 = torch.utils.data.DataLoader(
    test_1,
    batch_size=BATCH_SIZE, shuffle=False)
test_loader_2 = torch.utils.data.DataLoader(
    test_2,
    batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MiniModel().to(device)
model.load_state_dict(torch.load('assets/mini_classifier.pth'))
model.eval()

encodings_1 = np.empty((len(test_1), model.encode_dim))
encodings_2 = np.empty((len(test_2), model.encode_dim))

all_data = list()

with torch.no_grad():
    for loader, encodings in zip([test_loader_1, test_loader_2], [encodings_1, encodings_2]):
        for i, (data, _) in enumerate(loader):
            all_data.append(data.numpy())
            data = data.to(device)
            encoding = model.encode(data)
            encodings[i*loader.batch_size:i*loader.batch_size+data.shape[0]] = encoding.cpu().numpy()

all_data = np.concatenate(all_data, axis=0)

fid_lower = frechet_distance(
    encodings_1.mean(axis=0),
    np.cov(encodings_1, rowvar=False),
    encodings_2.mean(axis=0),
    np.cov(encodings_2, rowvar=False))

all_encodings = np.concatenate([encodings_1, encodings_2])
sampled_encodings_uniform = np.random.uniform(0, 1, size=all_data.shape)
sampled_encodings_normal = np.random.normal(0.1307, 0.3081, size=all_data.shape)
sampled_encodings_multivariate_normal = np.random.multivariate_normal(all_data.mean(axis=0), np.cov(all_data, rowvar=False), size=len(all_data))


def get_next_batch(data, batch_size):
    i = 0
    while i < len(data):
        yield torch.tensor(data[i:i+batch_size], dtype=torch.float32)
        i += batch_size


generator_uniform = get_next_batch(sampled_encodings_uniform, BATCH_SIZE)
generator_normal = get_next_batch(sampled_encodings_normal, BATCH_SIZE)
generator_multivariate_normal = get_next_batch(sampled_encodings_multivariate_normal, BATCH_SIZE)

fid_uniform = calculate_fid(lambda: next(generator_uniform), num_samples=len(all_data), verbose=False)
fid_normal = calculate_fid(lambda: next(generator_normal), num_samples=len(all_data), verbose=False)
fid_multivariate_normal = calculate_fid(lambda: next(generator_multivariate_normal), num_samples=len(all_data), verbose=False)

print(f'FID lower bound: {fid_lower:.4f}')
print(f'FID uniform: {fid_uniform:.4f}')
print(f'FID normal: {fid_normal:.4f}')
print(f'FID multivariate normal: {fid_multivariate_normal:.4f}')
