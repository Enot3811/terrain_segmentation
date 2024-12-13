"""Functions to work with torch DeeplabV3 model implementation."""


import torch
import torch.nn as nn


def create_deeplabv3_model(
    n_classes: int,
    model_name: str = 'deeplabv3_resnet50',
    pretrained: bool = True
) -> nn.Module:
    """Create DeeplabV3 model with custom number of classes.

    Parameters
    ----------
    n_classes : int
        Number of classes to use in the model.
    model_name : str, optional
        Name of the model to use. By default `'deeplabv3_resnet50'`.
    pretrained : bool, optional
        Whether to use pretrained weights. By default `True`.

    Returns
    -------
    nn.Module
        DeeplabV3 model with custom number of classes.

    Raises
    ------
    ValueError
        If the model name is not supported.
    """
    if model_name not in ['deeplabv3_resnet50',
                          'deeplabv3_resnet101',
                          'deeplabv3_mobilenet_v3_large']:
        raise ValueError(f'Model {model_name} is not supported.')

    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        model_name,
        weights='DEFAULT' if pretrained else None
    )

    # Replace last layer to change number of classes
    in_features = model.classifier[4].in_channels
    kernel_size = model.classifier[4].kernel_size
    stride = model.classifier[4].stride
    padding = model.classifier[4].padding
    dilation = model.classifier[4].dilation
    padding_mode = model.classifier[4].padding_mode
    bias = not model.classifier[4].bias is None

    model.classifier[4] = nn.Conv2d(
        in_features, n_classes, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, padding_mode=padding_mode,
        bias=bias
    )
    model.aux_classifier[4] = nn.Conv2d(
        in_features, n_classes, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, padding_mode=padding_mode,
        bias=bias
    )

    return model
