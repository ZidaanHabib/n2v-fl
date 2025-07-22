import torch
from torch import nn
from torch.nn import functional as F
from .double_conv import DoubleConv


class UNet(nn.Module): 
    def __init__(self, in_channels=1, base_out_channels=32, depth=4, batch_norm = True):
        super().__init__()

        # start with encoder portion of the network
        self.encoder_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            self.encoder_blocks.append(DoubleConv(channels,base_out_channels * 2**i, batch_norm))
            channels = base_out_channels * 2**i
        
        # downsamples will be handled in forward()

        self.bottom = DoubleConv(channels, channels * 2)
        channels = channels * 2 # tracking the number of channels at the bottom of the unet

        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList() # each up convolution has different params

        for i in reversed(range(depth)):
            self.up_convs.append(nn.ConvTranspose2d(channels,base_out_channels * 2**i, kernel_size=2, stride=2))
            channels = base_out_channels * 2**i
            self.decoder_blocks.append(DoubleConv(2*channels, channels, batch_norm)) 

        self.final_conv = nn.Conv2d(channels, in_channels,kernel_size=1)

    def forward(self, x: torch.Tensor): 
        skips = [] # skip connections that need to be copied from encoder to decoder
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x,2)

        # at the bottom of the Unet now
        x = self.bottom(x)

        for up, dec in zip(self.up_convs, self.decoder_blocks):
            x = up(x)
            skip = skips.pop()
            x = torch.cat((x,skip),dim=1)
            x = dec(x)
        
        return self.final_conv(x)



                                       