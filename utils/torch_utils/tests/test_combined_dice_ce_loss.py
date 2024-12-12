"""Tests for torch metrics implementations."""


import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.append(str(Path(__file__).parents[3]))
from utils.torch_utils.metrics import (
    CombinedDiceCrossEntropyLoss, DiceLoss)


def main():
    combined_loss = CombinedDiceCrossEntropyLoss(
        ce_weight=0.5,
        dice_weight=0.5,
        dice_activation='softmax'
    )
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    dice_loss = DiceLoss(activation='softmax', smooth=1e-6, reduction='mean')

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
    loss = combined_loss(predicts, one_hot_targets)

    seg_mask = one_hot_targets.argmax(dim=1)
    ce_loss_val = ce_loss(predicts, seg_mask)
    dice_loss_val = dice_loss(predicts, one_hot_targets)

    logger.info(f'Combined loss on perfect predictions: {loss}')
    logger.info(f'CE loss on perfect predictions: {ce_loss_val}')
    logger.info(f'Dice loss on perfect predictions: {dice_loss_val}')

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
    loss = combined_loss(predicts, one_hot_targets)

    seg_mask = one_hot_targets.argmax(dim=1)
    ce_loss_val = ce_loss(predicts, seg_mask)
    dice_loss_val = dice_loss(predicts, one_hot_targets)

    logger.info(f'Combined loss on completely wrong predictions: {loss}')
    logger.info(f'CE loss on completely wrong predictions: {ce_loss_val}')
    logger.info(f'Dice loss on completely wrong predictions: {dice_loss_val}')

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
             [1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]],

            [[1., 1., 1., 10.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [10., 1., 1., 1.]]
        ]
    ])  # shape: (1, 3, 4, 4))
    loss = combined_loss(predicts, one_hot_targets)

    seg_mask = one_hot_targets.argmax(dim=1)
    ce_loss_val = ce_loss(predicts, seg_mask)
    dice_loss_val = dice_loss(predicts, one_hot_targets)

    logger.info(f'Combined loss on perfect predictions: {loss}')
    logger.info(f'CE loss on intermediate predictions: {ce_loss_val}')
    logger.info(f'Dice loss on intermediate predictions: {dice_loss_val}')


if __name__ == '__main__':
    main()
