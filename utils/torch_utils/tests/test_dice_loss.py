"""Tests for torch metrics implementations."""


import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.append(str(Path(__file__).parents[3]))
from utils.torch_utils.metrics import DiceLoss


def main():
    dice_loss = DiceLoss(activation='softmax', smooth=1e-6, reduction='none')
    cls_weights = torch.tensor([0.0, 0.5, 1.0])
    weighted_dice_loss = DiceLoss(
        activation='softmax', smooth=1e-6, reduction='none',
        class_weights=cls_weights)
    logger.info(f'Class weights: {cls_weights}')

    # Test loss for perfect predictions.
    one_hot_targets = torch.tensor([
        [
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]],

            [[0., 1., 0., 0.],
             [1., 0., 1., 0.],
             [0., 1., 0., 1.],
             [0., 0., 1., 0.]],

            [[0., 0., 1., 1.],
             [0., 0., 0., 1.],
             [1., 0., 0., 0.],
             [1., 1., 0., 0.]]
        ]
    ])  # shape: (1, 3, 4, 4)
    predicts = torch.tensor([
        [
            [[10., 1., 1., 1.],
             [1., 10., 1., 1.],
             [1., 1., 10., 1.],
             [1., 1., 1., 10.]],

            [[1., 10., 1., 1.],
             [10., 1., 10., 1.],
             [1., 10., 1., 10.],
             [1., 1., 10., 1.]],

            [[1., 1., 10., 10.],
             [1., 1., 1., 10.],
             [10., 1., 1., 1.],
             [10., 10., 1., 1.]]
        ]
    ])  # shape: (1, 3, 4, 4))
    loss = dice_loss(predicts, one_hot_targets)
    weighted_loss = weighted_dice_loss(predicts, one_hot_targets)
    logger.info(f'Loss on perfect predictions: {loss}')
    logger.info(f'Weighted loss on perfect predictions: {weighted_loss}')

    # Test loss for completely wrong predictions.
    one_hot_targets = torch.tensor([
        [
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]],

            [[0., 1., 0., 0.],
             [1., 0., 1., 0.],
             [0., 1., 0., 1.],
             [0., 0., 1., 0.]],

            [[0., 0., 1., 1.],
             [0., 0., 0., 1.],
             [1., 0., 0., 0.],
             [1., 1., 0., 0.]]
        ]
    ])  # shape: (1, 3, 4, 4)
    predicts = torch.tensor([
        [
            [[1., 1., 10., 10.],
             [1., 1., 1., 10.],
             [10., 1., 1., 1.],
             [10., 10., 1., 1.]],

            [[10., 1., 1., 1.],
             [1., 10., 1., 1.],
             [1., 1., 10., 1.],
             [1., 1., 1., 10.]],

            [[1., 10., 1., 1.],
             [10., 1., 10., 1.],
             [1., 10., 1., 10.],
             [1., 1., 10., 1.]],
        ]
    ])  # shape: (1, 3, 4, 4))
    loss = dice_loss(predicts, one_hot_targets)
    weighted_loss = weighted_dice_loss(predicts, one_hot_targets)
    logger.info(f'Loss on completely wrong predictions: {loss}')
    logger.info(
        f'Weighted loss on completely wrong predictions: {weighted_loss}')

    # Test loss intermediate predictions.
    one_hot_targets = torch.tensor([
        [
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]],

            [[0., 1., 0., 0.],
             [1., 0., 1., 0.],
             [0., 1., 0., 1.],
             [0., 0., 1., 0.]],

            [[0., 0., 1., 1.],
             [0., 0., 0., 1.],
             [1., 0., 0., 0.],
             [1., 1., 0., 0.]]
        ]
    ])  # shape: (1, 3, 4, 4)
    predicts = torch.tensor([
        [
            [[10., 10., 10., 1.],
             [10., 10., 10., 10.],
             [10., 10., 10., 10.],
             [1., 10., 10., 10.]],

            [[1., 1., 1., 1.],
             [10., 1., 1., 1.],
             [1., 10., 1., 1.],
             [1., 1., 1., 1.]],

            [[1., 1., 1., 10.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [10., 1., 1., 1.]]
        ]
    ])  # shape: (1, 3, 4, 4))
    loss = dice_loss(predicts, one_hot_targets)
    weighted_loss = weighted_dice_loss(predicts, one_hot_targets)
    logger.info(f'Loss on intermediate predictions: {loss}')
    logger.info(f'Weighted loss on intermediate predictions: {weighted_loss}')


if __name__ == '__main__':
    main()
