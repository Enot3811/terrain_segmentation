"""Segmentation training script."""


from pathlib import Path
import shutil
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from torchmetrics import (
    Dice, JaccardIndex, Accuracy, MeanMetric, MetricCollection)

from unet.model import Unet
from deeplabv3.deeplabv3 import create_deeplabv3_model
from utils.torch_utils.datasets import SegmentationDataset
from utils.torch_utils.metrics import DiceLoss, CombinedDiceCrossEntropyLoss
from utils.torch_utils.functions import (
    SaveImagesSegCallback, ShowPredictCallback, get_metric_class)


def main(config_pth: Path):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Check config correctness
    if (config['train_dataset_params']['one_hot_encoding'] !=
            config['val_dataset_params']['one_hot_encoding']):
        raise ValueError('Train and val dataset params must have the same '
                         'one-hot encoding value.')
    elif (not config['train_dataset_params']['one_hot_encoding'] and
          config['loss_metric'] in ['dice', 'combined']):
        raise ValueError('Dice and combined loss metrics are supported only '
                         'with one-hot encoding.')
    elif Path(config['train_dir']).name != config_pth.stem:
        input(f'Specified train directory "{config["train_dir"]}" '
              'is not equal to config file name. '
              'Press enter to continue.')
    if config['model_name'] == 'deeplabv3':
        main_loss_weight = config.get('main_loss_weight', None)
        aux_loss_weight = config.get('aux_loss_weight', None)
        if not main_loss_weight or not aux_loss_weight:
            raise ValueError('Main and aux loss weights must be specified '
                             'for deeplabv3 model.')

    # Random
    torch.manual_seed(config['random_seed'])

    # Device
    if config['device'] == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])

    # Train dir
    train_dir = Path(config['train_dir'])
    tensorboard_dir = train_dir / 'tensorboard'
    ckpt_dir = train_dir / 'ckpts'
    if not config['continue_training']:
        if train_dir.exists():
            input(f'Specified directory "{str(train_dir)}" already exists. '
                  'If continue, this directory will be deleted. '
                  'Press enter to continue.')
            shutil.rmtree(train_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (config.get('callback_params', None) and
            config['callback_params'].get('save_images', None)):
        images_dir = train_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

    # Check and load checkpoint
    if config['continue_training']:
        checkpoint = torch.load(ckpt_dir / 'last_checkpoint.pth')
        model_params = checkpoint['model_state_dict']
        optim_params = checkpoint['optimizer_state_dict']
        lr_params = checkpoint['scheduler_state_dict']
        start_ep = checkpoint['epoch']
    else:
        model_params = None
        optim_params = None
        lr_params = None
        start_ep = 0

    # Get tensorboard
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Get transforms
    if config['train_transforms']:
        train_transforms = []
        if config['train_transforms']['crop_size']:
            train_transforms.append(
                A.RandomCrop(*config['train_transforms']['crop_size']))
        train_transforms = A.Compose(train_transforms)
    else:
        train_transforms = None
    if config['val_transforms']:
        val_transforms = []
        if config['val_transforms']['crop_size']:
            val_transforms.append(
                A.RandomCrop(*config['val_transforms']['crop_size']))
        val_transforms = A.Compose(val_transforms)
    else:
        val_transforms = None

    # Get datasets and loaders
    train_dset = SegmentationDataset(
        **config['train_dataset_params'], transforms=train_transforms)
    val_dset = SegmentationDataset(
        **config['val_dataset_params'], transforms=val_transforms)

    train_loader = DataLoader(train_dset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle_train'],
                              num_workers=config['num_workers'],
                              collate_fn=SegmentationDataset.collate_func)
    val_loader = DataLoader(val_dset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle_val'],
                            num_workers=config['num_workers'],
                            collate_fn=SegmentationDataset.collate_func)

    # Get callbacks
    if config.get('callback_params', None):
        steps_per_call = config['callback_params']['steps_per_call']
        callbacks = []
        if config['callback_params'].get('save_images', None):
            callbacks.append(SaveImagesSegCallback(
                save_dir=images_dir,
                cls_to_color=train_dset.id_to_color,
                **config['callback_params']['save_images']
            ))
        if config['callback_params'].get('show_predict', None):
            callbacks.append(ShowPredictCallback(
                cls_to_color=train_dset.id_to_color,
                **config['callback_params']['show_predict']
            ))
    else:
        callbacks = None

    # Get the model
    if config['model_name'] == 'unet':
        model = Unet(**config['model_params'])
    elif config['model_name'] == 'deeplabv3':
        model = create_deeplabv3_model(**config['model_params'])
    else:
        raise ValueError(f'Got unsupported model name: {config["model_name"]}')
    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    # Get loss function
    if config['loss_metric'] == 'cross_entropy':
        cls_weights = torch.tensor(config['ce_class_weights'],
                                   device=device)
        calculate_loss = nn.CrossEntropyLoss(weight=cls_weights)
    elif config['loss_metric'] == 'dice':
        calculate_loss = DiceLoss()
    elif config['loss_metric'] == 'combined':
        cls_weights = torch.tensor(config['ce_class_weights'],
                                   device=device)
        calculate_loss = CombinedDiceCrossEntropyLoss(
            ce_class_weights=cls_weights)
    elif config['loss_metric'] == 'binary_cross_entropy':
        cls_weights = torch.tensor(config['bce_class_weights'],
                                   device=device)
        calculate_loss = nn.BCEWithLogitsLoss(pos_weight=cls_weights)
    else:
        raise ValueError(
            f'Got unsupported loss metric: {config["loss_metric"]}')

    # Get the optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['start_lr'],
                               weight_decay=config['weight_decay'])
    else:
        raise ValueError(
            f'Got unsupported optimizer type {str(config["optimizer"])}')
    if optim_params:
        optimizer.load_state_dict(optim_params)

    # Get the scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['n_epoch'], eta_min=config['end_lr'],
        last_epoch=start_ep - 1)
    if lr_params:
        lr_scheduler.load_state_dict(lr_params)

    # Get metrics
    train_loss_metric = MeanMetric()
    val_loss_metric = MeanMetric()
    train_loss_metric.to(device=device)
    val_loss_metric.to(device=device)

    

    train_iou_metric = MeanMetric()
    train_dice_metric = MeanMetric()
    train_pixel_acc_metric = MeanMetric()
    train_iou_metric.to(device=device)
    train_dice_metric.to(device=device)
    train_pixel_acc_metric.to(device=device)

    val_iou_metric = MeanMetric()
    val_dice_metric = MeanMetric()
    val_pixel_acc_metric = MeanMetric()
    val_iou_metric.to(device=device)
    val_dice_metric.to(device=device)
    val_pixel_acc_metric.to(device=device)

    # Create aux metrics for deeplabv3
    if config['model_name'] == 'deeplabv3':
        train_aux_loss_metric = MeanMetric()
        val_aux_loss_metric = MeanMetric()
        train_aux_loss_metric.to(device=device)
        val_aux_loss_metric.to(device=device)

        train_aux_iou_metric = MeanMetric()
        train_aux_dice_metric = MeanMetric()
        train_aux_pixel_acc_metric = MeanMetric()
        train_aux_iou_metric.to(device=device)
        train_aux_dice_metric.to(device=device)
        train_aux_pixel_acc_metric.to(device=device)

        val_aux_iou_metric = MeanMetric()
        val_aux_dice_metric = MeanMetric()
        val_aux_pixel_acc_metric = MeanMetric()
        val_aux_iou_metric.to(device=device)
        val_aux_dice_metric.to(device=device)
        val_aux_pixel_acc_metric.to(device=device)

    # Do training
    best_metric = None
    for epoch in range(start_ep, config['n_epoch']):

        # Train epoch
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} train')
        for step, batch in enumerate(train_loader):
            images, masks, img_paths, mask_paths, shapes = batch
            images = images.to(device=device)
            out_logits = model(images)  # b, n_cls, h, w

            # Prepare masks for loss calculation
            if config['loss_metric'] == 'binary_cross_entropy':
                # BCE needs float32 masks
                masks = masks.to(dtype=torch.float32, device=device)
            else:
                masks = masks.to(device=device)

            # Different loss calculation for different models
            if config['model_name'] == 'unet':

                if config['loss_metric'] == 'cross_entropy':
                    loss = calculate_loss(out_logits, masks.argmax(dim=1))
                else:
                    loss = calculate_loss(out_logits, masks)
            
            else:  # deeplabv3

                aux_logits = out_logits['aux']
                out_logits = out_logits['out']
                if config['loss_metric'] == 'cross_entropy':
                    main_loss = calculate_loss(
                        out_logits, masks.argmax(dim=1))
                    aux_loss = calculate_loss(
                        aux_logits, masks.argmax(dim=1))
                else:
                    main_loss = calculate_loss(out_logits, masks)
                    aux_loss = calculate_loss(aux_logits, masks)
                loss = (main_loss * main_loss_weight +
                        aux_loss * aux_loss_weight)

            loss.backward()

            # Whether to update weights
            if (step % config['grad_accumulate_steps'] == 0 or
                    (step + 1 == len(train_loader))):
                optimizer.step()
                optimizer.zero_grad()

            # Calculate metrics
            with torch.no_grad():
                
                iou = iou_segmentation(
                    out_logits, masks, activation='softmax')
                dice = dice_coefficient(
                    out_logits, masks, activation='softmax')
                pixel_acc = pixel_accuracy(out_logits, masks)

                # Calculate aux metrics for deeplabv3
                if config['model_name'] == 'deeplabv3':
                    aux_iou = iou_segmentation(
                        aux_logits, masks, activation='softmax')
                    aux_dice = dice_coefficient(
                        aux_logits, masks, activation='softmax')
                    aux_pixel_acc = pixel_accuracy(aux_logits, masks)

            # Save metrics
            train_loss_metric.update(loss)
            train_iou_metric.update(iou)
            train_dice_metric.update(dice)
            train_pixel_acc_metric.update(pixel_acc)

            if config['model_name'] == 'deeplabv3':
                train_aux_loss_metric.update(aux_loss)
                train_aux_iou_metric.update(aux_iou)
                train_aux_dice_metric.update(aux_dice)
                train_aux_pixel_acc_metric.update(aux_pixel_acc)

            # Call callbacks
            if callbacks and (step % steps_per_call == 0 or
                              step + 1 == len(train_loader)):
                
                # Make predictions for callbacks
                if train_dset.binary_segmentation:
                    predicts = (out_logits.squeeze().detach().cpu().numpy() >
                                config['conf_threshold'])
                else:
                    predicts = out_logits.argmax(dim=1).detach().cpu().numpy()

                for callback in callbacks:
                    callback(batch, predicts, epoch, step)

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(f'Batch loss: {loss.item():.4f}')
        
        # Calculate mean metrics
        train_loss = train_loss_metric.compute()
        train_iou = train_iou_metric.compute()
        train_dice = train_dice_metric.compute()
        train_pixel_acc = train_pixel_acc_metric.compute()
        train_loss_metric.reset()
        train_iou_metric.reset()
        train_dice_metric.reset()
        train_pixel_acc_metric.reset()
        pbar.set_postfix_str(f'Epoch loss: {train_loss.item():.4f}')
        pbar.close()
        
        if config['model_name'] == 'deeplabv3':
            train_aux_loss = train_aux_loss_metric.compute()
            train_aux_iou = train_aux_iou_metric.compute()
            train_aux_dice = train_aux_dice_metric.compute()
            train_aux_pixel_acc = train_aux_pixel_acc_metric.compute()
            train_aux_loss_metric.reset()
            train_aux_iou_metric.reset()
            train_aux_dice_metric.reset()
            train_aux_pixel_acc_metric.reset()
        
        # Val epoch
        with torch.no_grad():
            model.eval()
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} val')
            for step, batch in enumerate(val_loader):
                images, masks, img_paths, mask_paths, shapes = batch
                images = images.to(device=device)
                out_logits = model(images)

                # Prepare masks for loss calculation
                if config['loss_metric'] == 'binary_cross_entropy':
                    # BCE needs float32 masks
                    masks = masks.to(dtype=torch.float32, device=device)
                else:
                    masks = masks.to(device=device)

                # Different loss calculation for different models
                if config['model_name'] == 'unet':

                    if config['loss_metric'] == 'cross_entropy':
                        loss = calculate_loss(out_logits, masks.argmax(dim=1))
                    else:
                        loss = calculate_loss(out_logits, masks)
                
                else:  # deeplabv3

                    aux_logits = out_logits['aux']
                    out_logits = out_logits['out']
                    if config['loss_metric'] == 'cross_entropy':
                        main_loss = calculate_loss(
                            out_logits, masks.argmax(dim=1))
                        aux_loss = calculate_loss(
                            aux_logits, masks.argmax(dim=1))
                    else:
                        main_loss = calculate_loss(out_logits, masks)
                        aux_loss = calculate_loss(aux_logits, masks)
                    loss = (main_loss * main_loss_weight +
                            aux_loss * aux_loss_weight)

                # Calculate metrics
                iou = iou_segmentation(
                    out_logits, masks, activation='softmax')
                dice = dice_coefficient(
                    out_logits, masks, activation='softmax')
                pixel_acc = pixel_accuracy(out_logits, masks)

                # Calculate aux metrics for deeplabv3
                if config['model_name'] == 'deeplabv3':
                    aux_iou = iou_segmentation(
                        aux_logits, masks, activation='softmax')
                    aux_dice = dice_coefficient(
                        aux_logits, masks, activation='softmax')
                    aux_pixel_acc = pixel_accuracy(aux_logits, masks)

                # Save metrics
                val_loss_metric.update(loss)
                val_iou_metric.update(iou)
                val_dice_metric.update(dice)
                val_pixel_acc_metric.update(pixel_acc)

                if config['model_name'] == 'deeplabv3':
                    val_aux_loss_metric.update(aux_loss)
                    val_aux_iou_metric.update(aux_iou)
                    val_aux_dice_metric.update(aux_dice)
                    val_aux_pixel_acc_metric.update(aux_pixel_acc)

                # Call callbacks
                if callbacks and (step % steps_per_call == 0 or
                                  step + 1 == len(val_loader)):
                    
                    # Make predictions for callbacks
                    if train_dset.binary_segmentation:
                        predicts = (
                            out_logits.squeeze().detach().cpu().numpy() >
                            config['conf_threshold'])
                    else:
                        predicts = (
                            out_logits.argmax(dim=1).detach().cpu().numpy())

                    for callback in callbacks:
                        callback(batch, predicts, epoch, step)

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(f'Batch loss: {loss.item():.4f}')

            # Calculate mean metrics
            val_loss = val_loss_metric.compute()
            val_iou = val_iou_metric.compute()
            val_dice = val_dice_metric.compute()
            val_pixel_acc = val_pixel_acc_metric.compute()
            val_loss_metric.reset()
            val_iou_metric.reset()
            val_dice_metric.reset()
            val_pixel_acc_metric.reset()
            pbar.set_postfix_str(f'Epoch loss: {val_loss.item():.4f}')
            pbar.close()

            if config['model_name'] == 'deeplabv3':
                val_aux_loss = val_aux_loss_metric.compute()
                val_aux_iou = val_aux_iou_metric.compute()
                val_aux_dice = val_aux_dice_metric.compute()
                val_aux_pixel_acc = val_aux_pixel_acc_metric.compute()
                val_aux_loss_metric.reset()
                val_aux_iou_metric.reset()
                val_aux_dice_metric.reset()
                val_aux_pixel_acc_metric.reset()

        # Lr scheduler
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log epoch metrics
        log_writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        log_writer.add_scalars('iou', {
            'train': train_iou,
            'val': val_iou
        }, epoch)
        log_writer.add_scalars('dice', {
            'train': train_dice,
            'val': val_dice
        }, epoch)
        log_writer.add_scalars('pixel_acc', {
            'train': train_pixel_acc,
            'val': val_pixel_acc
        }, epoch)

        if config['model_name'] == 'deeplabv3':
            log_writer.add_scalars('aux_loss', {
                'train': train_aux_loss,
                'val': val_aux_loss
            }, epoch)
            log_writer.add_scalars('aux_iou', {
                'train': train_aux_iou,
                'val': val_aux_iou
            }, epoch)
            log_writer.add_scalars('aux_dice', {
                'train': train_aux_dice,
                'val': val_aux_dice
            }, epoch)
            log_writer.add_scalars('aux_pixel_acc', {
                'train': train_aux_pixel_acc,
                'val': val_aux_pixel_acc
            }, epoch)

        log_writer.add_scalar('Lr', lr, epoch)

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        if best_metric is None or best_metric < val_loss.item():
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = val_loss.item()

    log_writer.close()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config_pth', type=Path,
        help='Path to train config.')
    args = parser.parse_args()

    if not args.config_pth.exists():
        raise FileNotFoundError(
            f'Config file "{str(args.config_pth)}" is not found.')
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    main(config_pth=config_pth)
