"""Module with helper functions for training."""

import yaml
from typing import Dict, Any
import albumentations as A


def get_transforms(transforms_conf: Dict[str, Any]) -> A.Compose:
    transforms = []
    for transform_conf in transforms_conf.values():
        if hasattr(A, transform_conf['class_name']):
            transform_class = getattr(A, transform_conf['class_name'])
            transform_params = transform_conf['params']
            transform = transform_class(**transform_params)
            transforms.append(transform)
        else:
            raise ValueError(
                f"Transform {transform_conf['class_name']} not found "
                "in albumentations")
    transforms = A.Compose(transforms)
    return transforms


def read_config(config_path: str) -> Dict[str, Any]:
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
    return config
