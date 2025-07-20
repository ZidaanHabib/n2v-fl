import torch
from torch import nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net  = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )           
    def forward(self, x: torch.Tensor):
        return self.net(x)     
