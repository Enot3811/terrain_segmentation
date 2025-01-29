"""Module with helper functions for training."""

import torchmetrics.segmentation
import yaml
from typing import Dict, Any, Optional, Tuple, Type, List, Union
from pathlib import Path
import shutil

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
import torchmetrics


def get_transforms(transforms_conf: Dict[str, Any]) -> Optional[A.Compose]:
    transforms = []
    for transform_conf in transforms_conf.values():
        if transform_conf['class_name'] == 'ToTensorV2':
            transforms.append(ToTensorV2())
        elif hasattr(A, transform_conf['class_name']):
            transform_class = getattr(A, transform_conf['class_name'])
            transform_params = transform_conf['params']
            transform = transform_class(**transform_params)
            transforms.append(transform)
        else:
            raise ValueError(
                f"Transform {transform_conf['class_name']} not found "
                "in albumentations")
    if len(transforms) > 0:
        return A.Compose(transforms)
    return None


def read_config(config_path: str) -> Dict[str, Any]:
    """Read and preprocess config.

    Parameters
    ----------
    config_path : str
        Path to config file.

    Returns
    -------
    Dict[str, Any]
        Preprocessed config.
    """
    # Read config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Process templates
    config['train_dataset_params']['dataset_path'] = (
        config['train_dataset_params']['dataset_path'].format(
            data_root=config['data_root']))
    config['val_dataset_params']['dataset_path'] = (
        config['val_dataset_params']['dataset_path'].format(
            data_root=config['data_root']))
    
    if config.get('callback', None):
        config['callback']['params']['save_dir'] = (
            config['callback']['params']['save_dir'].format(
                train_dir=config['train_dir']))
    
    return config


def get_smp_loss_fn(loss_config: Dict[str, Any]) -> torch.nn.Module:
    if loss_config['class_name'] == 'SMPDiceLossWrapper':
        loss_class = SMPDiceLossWrapper
        loss_params = loss_config['params']
        loss = loss_class(**loss_params)
        return loss
    elif hasattr(smp.losses, loss_config['class_name']):
        loss_class = getattr(smp.losses, loss_config['class_name'])
        loss_params = loss_config['params']
        loss = loss_class(**loss_params)
        return loss
    else:
        raise ValueError(f'Loss {loss_config["class_name"]} not found in '
                         'segmentation_models_pytorch')
    

def create_train_dir(config: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    train_dir = Path(config['train_dir'])
    tensorboard_dir = train_dir / 'tensorboard'
    ckpt_dir = train_dir / 'ckpts'
    if not config['continue_training']:
        if train_dir.exists():
            input(f'Specified directory "{str(train_dir)}" already exists. '
                  'Press enter to delete it and continue.')
            shutil.rmtree(train_dir)
        tensorboard_dir.mkdir(parents=True)
        ckpt_dir.mkdir(parents=True)
    return train_dir, tensorboard_dir, ckpt_dir


def get_model(model_config: Dict[str, Any]) -> torch.nn.Module:
    if hasattr(smp, model_config['class_name']):
        model_class = getattr(smp, model_config['class_name'])
        model_params = model_config['params']
        model = model_class(**model_params)
        return model
    else:
        raise ValueError(f'Model {model_config["class_name"]} not found in '
                         'segmentation_models_pytorch')
    

def get_metric_class(class_name: str) -> Type[torchmetrics.Metric]:
    """Get metric class from torchmetrics.

    Parameters
    ----------
    class_name : str
        Name of the metric class.

    Returns
    -------
    Type[torchmetrics.Metric]
        Metric class.

    Raises
    ------
    ValueError
        If metric class is not found in torchmetrics.
    """
    if class_name == 'DiceScore':
        return torchmetrics.segmentation.DiceScore
    elif hasattr(torchmetrics, class_name):
        return getattr(torchmetrics, class_name)
    raise ValueError(f"Metric class {class_name} not found in torchmetrics")


def create_metric_collection(
    metrics_config: Dict[str, Any]
) -> torchmetrics.MetricCollection:
    metrics = []
    for metric_config in metrics_config.values():
        metric_class = get_metric_class(metric_config['class_name'])
        metric_params = metric_config['params']
        metric = metric_class(**metric_params)
        metrics.append(metric)
    return torchmetrics.MetricCollection(metrics)


class SMPDiceLossWrapper(smp.losses.DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        reduction: Optional[str] = 'mean'
    ):
        """Wrapper for DiceLoss from segmentation_models_pytorch.

        This wrapper allows to weigh the loss for each class and to use
        user-defined reduction method.

        Parameters
        ----------
        mode : str
            Loss mode 'binary', 'multiclass' or 'multilabel'
        classes : Optional[List[int]], optional
            List of classes that contribute in loss computation
        log_loss : bool, optional
            If True, loss computed as `- log(dice_coeff)`
        from_logits : bool, optional
            If True, assumes input is raw logits
        smooth : float, optional
            Smoothness constant for dice coefficient
        ignore_index : Optional[int], optional
            Label that indicates ignored pixels
        eps : float, optional
            Small epsilon for numerical stability
        class_weights : Optional[torch.Tensor], optional
            Tensor of class weights (shape [C])
        reduction : Optional[str], optional
            Reduction method: 'mean', 'sum', 'none' or None

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps)

        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        if reduction in ['mean', 'sum', 'none', None]:
            self.reduction = reduction
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Overridden aggregation method.

        Weights the loss for each class and uses the defined reduction method.
        """
        # Put weights into aggregation step to avoid editing forward method
        loss = self.weigh_loss(loss)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
    def weigh_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Weights the loss for each class."""
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss
