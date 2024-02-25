import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniModel(nn.Module):

    def __init__(
        self,
        input_shape=(1, 28, 28),
        width=8,
        depth=4,
        dropout=0.5,
        num_classes=10
    ) -> None:
        super(MiniModel, self).__init__()
        """
        Inspired by miniVGG and Inception v3

        Args:
            input_shape (tuple, optional): Input shape. Defaults to (3, 28, 28).
            width (int, optional): Width of the first layer. Defaults to 8.
            depth (int, optional): Number of layers. Defaults to 3.
            num_classes (int, optional): Number of classes. Defaults to 10.
        """
        _w, _r1, _r2 = input_shape

        blocks = list()
        for _ in range(depth):
            blocks.append(self.create_block(_w, width))
            _w = width
            width *= 2
        self.blocks = nn.Sequential(*blocks)
        
        self.encode_dim = int(width/2)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encode_dim, num_classes)

    def create_block(self, in_width, out_width) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm2d(in_width),
            nn.Conv2d(in_width, out_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_width, out_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))
        x = self.blocks(x)
        x = self.average_pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.head(x)
        return x
