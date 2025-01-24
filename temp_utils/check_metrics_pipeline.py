"""Experiments with new metrics tracker.
"""


import sys
from pathlib import Path

import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parents[1]))
from utils.train_utils import (
    read_config, get_smp_loss_fn, create_metric_collection)


def get_test_cases():
    cases = []
    cases.append({
        'name': 'Perfect',
        'targets': torch.tensor([[
            [0, 1, 2, 2],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 2, 1, 0]]]
        ),
        'logits': torch.tensor([[
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
        ]])
    })

    cases.append({
        'name': 'Wrong',
        'targets': torch.tensor([[
            [0, 1, 2, 2],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 2, 1, 0]]]
        ),
        'logits': torch.tensor([[
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
                [1., 1., 10., 1.]]
        ]])
    })

    cases.append({
        'name': 'Intermediate',
        'targets': torch.tensor([[
            [0, 1, 2, 2],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 2, 1, 0]]]
        ),
        'logits': torch.tensor([[
            [
                [10., 10., 10., 1.],
                [1., 10., 10., 10.],
                [10., 1., 10., 10.],
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
        ]])
    })
    return cases


def check_separate_metrics():
    # Get test cases
    cases = get_test_cases()

    # Get metrics
    dice = torchmetrics.segmentation.DiceScore(
        num_classes=3, average=None, input_format='index')
    f1 = torchmetrics.F1Score(
        num_classes=3, average=None, task='multiclass')
    accuracy = torchmetrics.Accuracy(
        num_classes=3, average=None, task='multiclass')
    iou = torchmetrics.JaccardIndex(
        num_classes=3, average=None, task='multiclass')

    for case in cases:
        name = case['name']
        logits = case['logits']
        pred_indexes = logits.argmax(dim=1)
        seg_mask = case['targets']

        dice_values = dice(pred_indexes, seg_mask)
        f1_values = f1(pred_indexes, seg_mask)
        accuracy_values = accuracy(pred_indexes, seg_mask)
        iou_values = iou(pred_indexes, seg_mask)

        print(f'{name}:')
        print(f'Dice: {dice_values}')
        print(f'F1: {f1_values}')
        print(f'Accuracy: {accuracy_values}')
        print(f'IoU: {iou_values}')


def get_random_case(n_classes: int, b_size: int = 4):
    """Get random logits and seg_mask."""
    return (
        torch.rand(b_size, n_classes, 4, 4),
        torch.randint(0, n_classes, (b_size, 4, 4))
    )


@torch.no_grad()
def check_metrics_pipeline(config_path: Path, random_samples: bool = True):
    epochs = 10

    # Read config
    config = read_config(config_path)

    # Get metrics
    train_metrics = create_metric_collection(config['metrics'])
    val_metrics = create_metric_collection(config['metrics'])

    # Get loss
    loss_fn = get_smp_loss_fn(config['loss'])
    train_loss_metric = torchmetrics.MeanMetric()
    val_loss_metric = torchmetrics.MeanMetric()

    # Get tensorboard
    tensorboard_dir = Path('test_tensorboard')
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Get test cases
    if not random_samples:
        train_cases = get_test_cases()
        val_cases = train_cases[1:] + train_cases[:1]

    for epoch in range(epochs):

        for _ in range(5):
            if random_samples:
                logits, seg_mask = get_random_case(config['num_classes'])
            else:
                logits = train_cases[epoch % len(train_cases)]['logits']
                seg_mask = train_cases[epoch % len(train_cases)]['targets']

            loss = loss_fn(logits, seg_mask)
            train_loss_metric.update(loss)

            predicts = logits.argmax(dim=1)
            train_metrics(predicts, seg_mask)

        for _ in range(5):
            if random_samples:
                logits, seg_mask = get_random_case(config['num_classes'])
            else:
                logits = val_cases[epoch % len(val_cases)]['logits']
                seg_mask = val_cases[epoch % len(val_cases)]['targets']

            loss = loss_fn(logits, seg_mask)
            val_loss_metric.update(loss)

            predicts = logits.argmax(dim=1)
            val_metrics(predicts, seg_mask)

        # Log metrics
        train_loss = train_loss_metric.compute()
        val_loss = val_loss_metric.compute()
        train_loss_metric.reset()
        val_loss_metric.reset()
        log_writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        train_metric_values = train_metrics.compute()
        val_metric_values = val_metrics.compute()
        train_metrics.reset()
        val_metrics.reset()

        for metric_name in train_metrics:

            # Log mean values
            log_writer.add_scalars(metric_name, {
                'train': train_metric_values[metric_name].mean(),
                'val': val_metric_values[metric_name].mean(),
            }, epoch)

            # Log class values
            log_writer.add_scalars(metric_name + '_per_class', {
                f'train_class_{i}': value
                for i, value in enumerate(train_metric_values[metric_name])
            }, epoch)
            log_writer.add_scalars(metric_name + '_per_class', {
                f'val_class_{i}': value
                for i, value in enumerate(val_metric_values[metric_name])
            }, epoch)

    log_writer.close()


if __name__ == '__main__':
    # check_separate_metrics()
    config_path = Path('train_configs/test_config.yaml')
    check_metrics_pipeline(config_path)
