"""U-Net model from the paper https://arxiv.org/pdf/1505.04597.pdf."""

from typing import List
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parents[2]))
from .crop_concat import CropConcat


class Unet(nn.Module):
    def __init__(
        self, output_channels: int, n_stages: int = 5,
        initial_channels: int = 64, channels_upsampling: int = 2,
        image_channels: int = 3, use_pad: bool = False
    ):
        """Initialize U-Net with specified parameters.

        Parameters
        ----------
        output_channels : int
            A number of channels in output feature map.
        n_stages : int, optional
            A number of levels in the net. By default is 5.
        initial_channels : int, optional
            A number of channels after initial convolution
            as well as the base for channel upsampling. By default is 64.
        channels_upsampling : int, optional
            A coefficient of the channel upsampling after each stage.
            Formula: `channels_count = base * upsample ** stage_i`.
        image_channels : int, optional
            A number of input image's channels. By default is 3.
        use_pad : bool, optional
            Whether convolution layers use padding to keep feature map's
            size the same. By default `False`.
        """
        super().__init__()
        self._config = {
            'output_channels': output_channels,
            'n_stages': n_stages,
            'initial_channels': initial_channels,
            'initial_channels': initial_channels,
            'channels_upsampling': channels_upsampling,
            'image_channels': image_channels,
            'use_pad': use_pad}

        padding = 1 if use_pad else 0
        in_ch = image_channels
        out_ch = initial_channels
        self.left_side = nn.ModuleDict()
        for i in range(n_stages - 1):
            self.left_side.add_module(
                name=f'encoder{i + 1}',
                module=nn.Sequential(
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size=3, padding=padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch,
                              kernel_size=3, padding=padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)))
            in_ch = out_ch
            out_ch *= channels_upsampling

            self.left_side.add_module(
                name=f'pool{i + 1}',
                module=nn.MaxPool2d(kernel_size=2, stride=2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.crop_concat = CropConcat()
        
        in_ch = out_ch
        out_ch //= 2
        self.right_side = nn.ModuleDict()
        for i in range(n_stages - 1):
            self.right_side.add_module(
                name=f'upconv{i + 1}',
                module=nn.ConvTranspose2d(
                    in_ch, out_ch, kernel_size=2, stride=2))
            
            self.right_side.add_module(
                name=f'decoder{i + 1}',
                module=nn.Sequential(
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size=3, padding=padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch,
                              kernel_size=3, padding=padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)))
            in_ch = out_ch
            out_ch //= 2
        
        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call U-Net on batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Data tensor of shape `(b, img_c, h, w)`.

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape `(b, out_c, out_h, out_w)`.
        """
        skip_cons: List[torch.Tensor] = []
        for layer in self.left_side.values():
            x = layer(x)
            # If conv
            if not isinstance(layer, nn.MaxPool2d):
                # Save skip connection
                skip_cons.append(x)

        x = self.bottleneck(x)

        for layer in self.right_side.values():
            x = layer(x)
            # Concat after ConvTranspose
            if isinstance(layer, nn.ConvTranspose2d):
                # Get skip connection
                x = self.crop_concat(x, skip_cons.pop())

        x = self.final_conv(x)
        return x
    
    def get_config(self):
        """Get config from which model can be created."""
        return self._config

    @classmethod
    def from_config(cls, config) -> 'Unet':
        """Create model instance from its config."""
        return cls(**config)


if __name__ == '__main__':
    c = 3
    h = w = 640
    b_size = 4
    test_data = torch.rand((b_size, c, h, w))

    out_ch = 1
    model = Unet(out_ch, initial_channels=64, use_pad=True)
    out = model(test_data)
    print("Model's scheme:", model, sep='\n')
    print('Input shape:', test_data.shape)
    print('Output shape:', out.shape)
