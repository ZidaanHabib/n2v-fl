import torch
from torch import nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.net  = nn.Sequential(*layers)           

    def forward(self, x: torch.Tensor):
        return self.net(x)     
