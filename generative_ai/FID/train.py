import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

from model import MiniModel

BATCH_SIZE = 128
EPOCHS = 50
FRAC_TRAIN = 0.9
EARLY_STOPPING = 10


_transforms = transforms.Compose ([
    transforms.ToTensor (),
    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
    transforms.Lambda(lambda x: x.flatten())
])

train_dataset, val_dataset = torch.utils.data.random_split(
    datasets.MNIST('data', train=True, download=True, transform=_transforms),
    (FRAC_TRAIN, 1 - FRAC_TRAIN),
    generator=torch.Generator().manual_seed(42)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=_transforms),
    batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MiniModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0
best_epoch = 0

for epoch in tqdm.trange(EPOCHS, desc='Epochs'):

    model.train()
    for data, target in tqdm.tqdm(train_loader, desc='train loop', leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        for data, target in tqdm.tqdm(val_loader, desc='val loop', leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        print(f'Epoch {epoch}: val_loss: {val_loss:.4f}, accuracy: {accuracy:.2f}%')
        
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), 'assets/mini_classifier.pth')
    
    if epoch - best_epoch > EARLY_STOPPING:
        print(f'Early stopping at epoch {epoch}')
        break

model.load_state_dict(torch.load('assets/mini_classifier.pth'))
model.eval()

test_loss = 0
correct = 0

encodings = np.empty((len(test_loader.dataset), model.encode_dim))
with torch.no_grad():
    for idx, (data, target) in enumerate(tqdm.tqdm(test_loader, desc='test loop')):
        data, target = data.to(device), target.to(device)
        encoding = model.encode(data)
        encodings[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE] = encoding.cpu().numpy()
        output = model.head(encoding)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

np.save('assets/test_encodings_for_fid.npy', encodings)

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'\nTest set: test_loss: {test_loss:.4f}, accuracy: {accuracy:.2f}%')
