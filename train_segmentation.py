"""Segmentation training script."""


from pathlib import Path
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics import MeanMetric

from utils.torch_utils.datasets import SegmentationDataset
from utils.torch_utils.functions import SaveImagesSegCallback
from utils.train_utils import (
    get_transforms, read_config, create_train_dir, get_model, get_smp_loss_fn,
    create_metric_collection)


def main(config_pth: Path):
    # Read config
    config = read_config(config_pth)

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
    train_dir, tensorboard_dir, ckpt_dir = create_train_dir(config)

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
    train_transforms = get_transforms(config['train_transforms'])
    val_transforms = get_transforms(config['val_transforms'])

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

    # Get callback
    if config.get('callback', None):
        callback = SaveImagesSegCallback(**config['callback']['params'])
    else:
        callback = None

    # Get the model
    model = get_model(config['model'])
    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    # Get loss function
    # TODO: smp losses does not support class weights
    loss_fn = get_smp_loss_fn(config['loss'])

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
    train_metrics = create_metric_collection(config['train_metrics']).to(
        device=device)
    val_metrics = create_metric_collection(config['val_metrics']).to(
        device=device)
    train_loss_metric = MeanMetric().to(device=device)
    val_loss_metric = MeanMetric().to(device=device)

    # Do training
    best_metric = None
    for epoch in range(start_ep, config['n_epoch']):

        # Train epoch
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} train')
        for step, batch in enumerate(train_loader):
            images, masks, img_paths, mask_paths, shapes = batch
            images = images.to(device=device)
            masks = masks.to(device=device)
            out_logits = model(images)  # b, n_cls, h, w

            loss = loss_fn(out_logits, masks)
            loss.backward()

            # Whether to update weights
            if (step % config['grad_accumulate_steps'] == 0 or
                    (step + 1 == len(train_loader))):
                optimizer.step()
                optimizer.zero_grad()

            # Calculate metrics
            with torch.no_grad():
                train_metrics.update(out_logits, masks)
                train_loss_metric.update(loss)

            # Call callbacks
            if callback and (step % config['steps_per_call'] == 0 or
                             step + 1 == len(train_loader)):
                callback(batch, out_logits, epoch, step)

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(f'Batch loss: {loss.item():.4f}')
        
        # Calculate mean epoch loss
        train_loss = train_loss_metric.compute()
        train_metrics.reset()
        train_loss_metric.reset()
        pbar.set_postfix_str(f'Epoch loss: {train_loss.item():.4f}')
        pbar.close()
        
        # Val epoch
        with torch.no_grad():
            model.eval()
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} val')
            for step, batch in enumerate(val_loader):
                images, masks, img_paths, mask_paths, shapes = batch
                images = images.to(device=device)
                masks = masks.to(device=device)
                out_logits = model(images)

                loss = loss_fn(out_logits, masks)

                # Calculate metrics
                val_metrics.update(out_logits, masks)
                val_loss_metric.update(loss)

                # Call callbacks
                if callback and (step % config['steps_per_call'] == 0 or
                                 step + 1 == len(val_loader)):
                    callback(batch, out_logits, epoch, step)

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(f'Batch loss: {loss.item():.4f}')

            # Calculate mean epoch loss
            val_loss = val_loss_metric.compute()
            val_metrics.reset()
            val_loss_metric.reset()
            pbar.set_postfix_str(f'Epoch loss: {val_loss.item():.4f}')
            pbar.close()

        # Lr scheduler
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log epoch metrics
        log_writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        # Iterate over train and val metrics
        for (name, metric_train), metric_val in zip(
                train_metrics.items(), val_metrics.values()):
            # Compute each separately
            metric_train_values = metric_train.compute()
            metric_val_values = metric_val.compute()

            # Log mean values
            log_writer.add_scalars(name, {
                'train_avg': metric_train_values.mean(),
                'val_avg': metric_val_values.mean()
            }, epoch)

            # Log class values
            log_writer.add_scalars(name, {
                f'train_class_{i}': value
                for i, value in enumerate(metric_train_values)
            }, epoch)
            log_writer.add_scalars(name, {
                f'val_class_{i}': value
                for i, value in enumerate(metric_val_values)
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
