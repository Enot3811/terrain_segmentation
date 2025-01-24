import segmentation_models_pytorch as smp
import torch


# Define losses
reduction = None
dice_loss = smp.losses.DiceLoss(mode='multiclass')
cross_entropy_loss = smp.losses.SoftCrossEntropyLoss(
    reduction=reduction, smooth_factor=0.0)
focal_loss = smp.losses.FocalLoss(
    mode='multiclass',
    reduction=reduction,
    alpha=0.25,
    gamma=2.0
)

# Test loss for perfect predictions.
one_hot_targets = torch.tensor([
    [
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]],

        [
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.]],

        [
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [1., 1., 0., 0.]]
    ]
])  # shape: (1, 3, 4, 4)
targets = one_hot_targets.argmax(dim=1)
predicts = torch.tensor([
    [
        [
            [10., 1., 1., 1.],
            [1., 10., 1., 1.],
            [1., 1., 10., 1.],
            [1., 1., 1., 10.]],

        [
            [1., 10., 1., 1.],
            [10., 1., 10., 1.],
            [1., 10., 1., 10.],
            [1., 1., 10., 1.]],

        [
            [1., 1., 10., 10.],
            [1., 1., 1., 10.],
            [10., 1., 1., 1.],
            [10., 10., 1., 1.]]
    ]
])  # shape: (1, 3, 4, 4))
dice = dice_loss(predicts, targets)
cross_entropy = cross_entropy_loss(predicts, targets)
focal = focal_loss(predicts, targets)
print(f'Dice loss: {dice}')
print(f'Cross entropy loss: {cross_entropy}')
print(f'Focal loss: {focal}\n')

# Test loss for completely wrong predictions.
one_hot_targets = torch.tensor([
    [
        [[
            1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]],

        [
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.]],

        [
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [1., 1., 0., 0.]]
    ]
])  # shape: (1, 3, 4, 4)
predicts = torch.tensor([
    [
        [
            [1., 1., 10., 10.],
            [1., 1., 1., 10.],
            [10., 1., 1., 1.],
            [10., 10., 1., 1.]],

        [
            [10., 1., 1., 1.],
            [1., 10., 1., 1.],
            [1., 1., 10., 1.],
            [1., 1., 1., 10.]],

        [
            [1., 10., 1., 1.],
            [10., 1., 10., 1.],
            [1., 10., 1., 10.],
            [1., 1., 10., 1.]],
    ]
])  # shape: (1, 3, 4, 4))
dice = dice_loss(predicts, targets)
cross_entropy = cross_entropy_loss(predicts, targets)
focal = focal_loss(predicts, targets)
print(f'Dice loss: {dice}')
print(f'Cross entropy loss: {cross_entropy}')
print(f'Focal loss: {focal}\n')

# Test loss intermediate predictions.
one_hot_targets = torch.tensor([
    [
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]],

        [
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.]],

        [
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [1., 1., 0., 0.]]
    ]
])  # shape: (1, 3, 4, 4)
predicts = torch.tensor([
    [
        [
            [10., 10., 10., 1.],
            [10., 10., 10., 10.],
            [10., 10., 10., 10.],
            [1., 10., 10., 10.]],

        [
            [1., 1., 1., 1.],
            [10., 1., 1., 1.],
            [1., 10., 1., 1.],
            [1., 1., 1., 1.]],

        [
            [1., 1., 1., 10.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [10., 1., 1., 1.]]
    ]
])  # shape: (1, 3, 4, 4))
dice = dice_loss(predicts, targets)
cross_entropy = cross_entropy_loss(predicts, targets)
focal = focal_loss(predicts, targets)
print(f'Dice loss: {dice}')
print(f'Cross entropy loss: {cross_entropy}')
print(f'Focal loss: {focal}\n')
