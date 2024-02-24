import torch

from ddpm.unet import Unet
from ddpm.ddpm import DDPM
from FID.calculate_fid import calculate_fid


BATCH_SIZE = 128
NUM_SAMPLES = 10_000
T = 1000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

network = Unet().to(device)
ddpm_model = DDPM(network, T=T).to(device)
ddpm_model.load_state_dict(torch.load('assets/ddpm_model.pt', map_location=device))


sample_model = lambda: ddpm_model.sample((BATCH_SIZE, 28**2))

fid = calculate_fid(sample_model, num_samples=NUM_SAMPLES)

print(f'FID: {fid:.4f}')
