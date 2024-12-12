"""The crop and concatenate skip connection from U-Net.

This layer get feature map `x1` with shape `(b, c, h1, w1)` and `x2` with shape
`(b, c, h2, w2)` and `h1 < h2, w1 < w2`.
From `x2` a center crop is made and `x2_crop` with shape `(b, c, h1, w1)`
is obtained.
Then `x1` and `x2_crop` are concatenated and `out` with shape `(b, 2c, h1, w1)`
is obtained.
"""

import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop


class CropConcat(nn.Module):
    """The crop and concatenate skip connection from U-Net.

    This layer get features `x1` with shape `(b, c, h1, w1)` and `x2`
    with shape `(b, c, h2, w2)` and `h1 < h2, w1 < w2`.
    From `x2` a center crop is made and `x2_crop` with shape `(b, c, h1, w1)`
    is obtained.
    Then `x1` and `x2_crop` are concatenated
    and `out` with shape `(b, 2c, h1, w1)` is obtained.
    """

    def __init__(self) -> None:
        """Initialize CropConcat layer."""
        super().__init__()

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """Call CropConcat layer of passed inputs.

        Parameters
        ----------
        x1 : torch.Tensor
            A smaller feature map with shape `(b, c, h1, w1)`.
        x2 : torch.Tensor
            A larger feature map with shape `(b, c, h2, w2)`.

        Returns
        -------
        torch.Tensor
            Cropped and concatenated feature map with shape `(b, 2c, h, w)`
        """
        x2_crop = center_crop(x2, x1.shape[-2::1])
        return torch.cat((x1, x2_crop), dim=1)
