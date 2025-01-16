"""Metrics for PyTorch."""


from typing import List, Optional

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torchvision.ops import box_iou
from torchmetrics import Metric
from loguru import logger

from .functions import convert_seg_mask_to_one_hot


def iou_detection(
    gt_boxes: FloatTensor, predicted_boxes: List[FloatTensor]
) -> FloatTensor:
    """Calculate IoU over predictions and ground truth bounding boxes.

    Parameters
    ----------
    gt_boxes : FloatTensor
        Ground truth bounding boxes with shape `(n_img, n_max_obj, 4)`.
    predicted_boxes : List[FloatTensor]
        Proposals list with length `n_img`
        and each element has shape `(n_props_per_img, 4)`.

    Returns
    -------
    FloatTensor
        Calculated IoU scalar value.
    """
    # Calculate n objects in gt boxes
    n_objs = (gt_boxes >= 0).any(dim=2).sum(dim=1)
    iou_list = []
    for i, n_obj in enumerate(n_objs):
        iou = box_iou(gt_boxes[i][:n_obj], predicted_boxes[i])
        # Check is there a prediction
        if iou.shape[0] != 0 and iou.shape[1] != 0:
            iou, _ = iou.max(dim=1)
            iou = torch.mean(iou)
        else:
            iou = torch.zeros((1,), dtype=torch.float32, device=iou.device)
        iou_list.append(iou)
    iou = sum(iou_list) / len(iou_list)
    return iou


def iou_segmentation(
    predicted_logits: FloatTensor,
    targets: LongTensor,
    activation: str = 'sigmoid',
    sigmoid_threshold: float = 0.5,
    smooth: float = 1e-6
) -> FloatTensor:
    """Calculate IoU over predictions and ground truth masks.

    Parameters
    ----------
    predicted_logits : FloatTensor
        Predicted logits with shape `(b, n_classes, h, w)`.
    targets : LongTensor
        One-hot encoded targets with shape `(b, n_classes, h, w)`.
    activation : str, optional
        Activation function to apply to predictions.
        Options: `'sigmoid'`, `'softmax'`.
        By default, `'sigmoid'`.
    sigmoid_threshold : float, optional
        Threshold for sigmoid activation. If `activation` is `'softmax'`,
        this parameter is ignored. By default, `0.5`.
    smooth : float, optional
        Smoothing factor to avoid division by zero.
        By default, `1e-6`.

    Returns
    -------
    FloatTensor
        Calculated IoU scalar value.

    Raises
    ------
    ValueError
        If unsupported activation function is provided.
    """
    if activation == 'sigmoid':
        predictions = (
            torch.sigmoid(predicted_logits) > sigmoid_threshold).float()
    elif activation == 'softmax':
        predictions = torch.softmax(predicted_logits, dim=1).argmax(dim=1)
        predictions = convert_seg_mask_to_one_hot(
            predictions, targets.shape[1])
    else:
        raise ValueError(f'Unsupported activation function: {activation}')
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def dice_coefficient(
    predicted_logits: FloatTensor,
    targets: LongTensor,
    activation: str = 'sigmoid',
    sigmoid_threshold: float = 0.5,
    smooth: float = 1e-6
) -> FloatTensor:
    """Calculate Dice coefficient over predictions and ground truth masks.

    Parameters
    ----------
    predicted_logits : FloatTensor
        Predicted logits with shape `(b, n_classes, h, w)`.
    targets : LongTensor
        One-hot encoded targets with shape `(b, n_classes, h, w)`.
    activation : str, optional
        Activation function to apply to predictions.
        Options: `'sigmoid'`, `'softmax'`.
        By default, `'sigmoid'`.
    sigmoid_threshold : float, optional
        Threshold for sigmoid activation. If `activation` is `'softmax'`,
        this parameter is ignored. By default, `0.5`.
    smooth : float, optional
        Smoothing factor to avoid division by zero.
        By default, `1e-6`.

    Returns
    -------
    FloatTensor
        Calculated Dice coefficient scalar value.

    Raises
    ------
    ValueError
        If unsupported activation function is provided.
    """
    if activation == 'sigmoid':
        predictions = (
            torch.sigmoid(predicted_logits) > sigmoid_threshold).float()
    elif activation == 'softmax':
        predictions = torch.softmax(predicted_logits, dim=1).argmax(dim=1)
        predictions = convert_seg_mask_to_one_hot(
            predictions, targets.shape[1])
    else:
        raise ValueError(f'Unsupported activation function: {activation}')
    
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )
    return dice


def pixel_accuracy(
    predicted_logits: FloatTensor,
    targets: LongTensor,
    activation: str = 'sigmoid',
    sigmoid_threshold: float = 0.5
) -> FloatTensor:
    """Calculate pixel accuracy over predictions and ground truth masks.

    Parameters
    ----------
    predicted_logits : FloatTensor
        Predicted logits with shape `(b, n_classes, h, w)`.
    targets : LongTensor
        One-hot encoded targets with shape `(b, n_classes, h, w)`.
    activation : str, optional
        Activation function to apply to predictions.
        Options: `'sigmoid'`, `'softmax'`.
        By default, `'sigmoid'`.
    sigmoid_threshold : float, optional
        Threshold for sigmoid activation. If `activation` is `'softmax'`,
        this parameter is ignored. By default, `0.5`.

    Returns
    -------
    FloatTensor
        Calculated pixel accuracy scalar value.

    Raises
    ------
    ValueError
        If unsupported activation function is provided.
    """
    if activation == 'sigmoid':
        predictions = (
            torch.sigmoid(predicted_logits) > sigmoid_threshold).float()
    elif activation == 'softmax':
        predictions = torch.softmax(predicted_logits, dim=1).argmax(dim=1)
        predictions = convert_seg_mask_to_one_hot(
            predictions, targets.shape[1])
    else:
        raise ValueError(f'Unsupported activation function: {activation}')
    correct = (predictions == targets).float().sum()
    total = torch.numel(predictions)
    return correct / total


class CrossEntropyLossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.add_state('loss',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_total',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        
    def update(self, logits: FloatTensor, targets: LongTensor):
        loss = self.loss(logits, targets)
        self.loss += loss
        self.n_total += 1

    def compute(self):
        return self.loss / self.n_total


class DiceLoss(nn.Module):
    def __init__(
        self,
        activation: str = 'softmax',
        reduction: str = 'mean',
        class_weights: Optional[FloatTensor] = None,
        smooth: float = 1e-6
    ):
        """Initialize Dice loss.

        Parameters
        ----------
        activation : str, optional
            Activation function to apply to predictions.
            Options: `'softmax'`, `'sigmoid'`.
            By default, `'softmax'`.
        reduction : str, optional
            Reduction method for the loss.
            Options: `'mean'`, `'sum'`, `'none'`. If `'none'` classwise loss
            will be returned. By default, `'mean'`.
        class_weights : Optional[FloatTensor], optional
            Class weights for loss.
            By default, `None`.
        smooth : float, optional
            Smoothing factor to avoid division by zero.
            By default, `1e-6`.
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Unsupported reduction: "{reduction}"')
        self.activation = activation

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: FloatTensor,
        targets: LongTensor,
    ) -> FloatTensor:
        """Calculate Dice loss.

        Parameters
        ----------
        predictions : FloatTensor
            Predictions logits tensor with shape `(B, C, H, W)`.
        targets : LongTensor
            One-hot encoded targets tensor with shape `(B, C, H, W)`.

        Returns
        -------
        FloatTensor
            Calculated Dice loss.
        """
        if self.activation == 'softmax':
            predictions = torch.softmax(predictions, dim=1)
        elif self.activation == 'sigmoid':
            predictions = torch.sigmoid(predictions)
            # TODO: fix sigmoid activation
            logger.warning('Sigmoid activation is not working properly.')
        else:
            raise ValueError(
                f'Unsupported activation function: {self.activation}')
        
        # Calculate Dice for each class
        dice_per_class = []
        for channel in range(predictions.shape[1]):
            pred_channel = predictions[:, channel, ...]
            target_channel = targets[:, channel, ...]
            
            intersection = (pred_channel * target_channel).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_channel.sum() + target_channel.sum() + self.smooth
            )

            # Convert dice coefficient to loss and apply class weight
            loss = 1.0 - dice
            if self.class_weights is not None:
                loss = loss * self.class_weights[channel]
                
            dice_per_class.append(loss)
            
        class_losses = torch.stack(dice_per_class)
        
        if self.reduction == 'mean':
            return class_losses.mean()
        elif self.reduction == 'sum':
            return class_losses.sum()
        else:  # reduction is "none"
            return class_losses


class CombinedDiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ce_class_weights: Optional[FloatTensor] = None,
        dice_class_weights: Optional[FloatTensor] = None
    ):
        """Initialize combined Dice and Cross Entropy loss.

        Parameters
        ----------
        ce_weight : float, optional
            Weight of Cross Entropy loss.
            By default, `0.5`.
        dice_weight : float, optional
            Weight of Dice loss.
            By default, `0.5`.
        ce_class_weights : Optional[FloatTensor], optional
            Class weights for Cross Entropy loss.
            By default, `None`.
        dice_class_weights : Optional[FloatTensor], optional
            Class weights for Dice loss.
            By default, `None`.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.dice = DiceLoss(class_weights=dice_class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
    def forward(
        self,
        predictions: FloatTensor,
        targets: LongTensor
    ) -> FloatTensor:
        """Calculate combined Dice and Cross Entropy loss.

        Parameters
        ----------
        predictions : FloatTensor
            Predictions logits tensor with shape `(B, C, H, W)`.
        targets : LongTensor
            One-hot encoded targets tensor with shape `(B, C, H, W)`.

        Returns
        -------
        FloatTensor
            Calculated combined Dice and Cross Entropy loss.
        """
        # For CrossEntropyLoss, need targets as (B, H, W) with class indices
        targets_ce = targets.argmax(dim=1)
        ce_loss = self.ce(predictions, targets_ce)
        dice_loss = self.dice(predictions, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
