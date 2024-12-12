"""Script for UNet inference on images."""

import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import sys

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from unet.model import Unet
from utils.data_utils.functions import (
    collect_paths, IMAGE_EXTENSIONS, read_volume, show_images_cv2,
    resize_image)
from utils.torch_utils.datasets import SegmentationDataset
from utils.argparse_utils import natural_int


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model input."""
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, 0)
    return torch.from_numpy(image)


def postprocess_mask(
    mask: torch.Tensor,
    cls_to_color: Dict[str, List[int]]
) -> np.ndarray:
    """Convert model output to final segmentation mask."""
    # Apply softmax to get probabilities
    probs = torch.softmax(mask, dim=0)
    # Get class indices
    mask = torch.argmax(probs, dim=0)
    # Convert to numpy
    mask = mask.cpu().numpy()
    # Convert to color mask
    color_mask = SegmentationDataset.seg_mask_to_color(mask, cls_to_color)
    return color_mask


def main(
    samples_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    device: str = 'cuda',
    masks_path: Optional[Path] = None,
    resize_to_show: Optional[Tuple[int, int]] = None
) -> None:

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    cls_to_color = {
        cls_id: color
        for cls_id, color in zip(config['class_to_id'].values(),
                                 config['class_to_color'].values())}
    binary_seg = config['train_dataset_params'].get(
        'binary_segmentation', False)
    n_classes = len(cls_to_color) - int(binary_seg)

    # Initialize model
    model = Unet(**config['model_params'], output_channels=n_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Collect image paths
    if samples_path.is_file():
        image_paths = [samples_path]
    else:
        image_paths = sorted(
            collect_paths(samples_path, IMAGE_EXTENSIONS + ['npy']))
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {str(samples_path)}")
        
    # Collect mask paths
    if masks_path is not None:
        if masks_path.is_file():
            mask_paths = [masks_path]
        else:
            mask_paths = sorted(
                collect_paths(masks_path, IMAGE_EXTENSIONS + ['npy']))
        if len(mask_paths) == 0:
            raise ValueError(f"No masks found in {str(masks_path)}")

    # Process images
    for i, image_path in enumerate(
        tqdm(image_paths, desc="Processing images")
    ):
        # Load and preprocess image
        image = read_volume(image_path)
        tensor = preprocess_image(image)
        tensor = tensor.to(device)

        # Get prediction
        with torch.no_grad():
            output = model(tensor)[0]

        # Postprocess mask
        mask = postprocess_mask(output, cls_to_color)

        # Resize images if needed
        if resize_to_show is not None:
            image = resize_image(image, resize_to_show)
            mask = resize_image(mask, resize_to_show)

        images_to_show = [image, mask]
        titles = ['Image', 'Prediction']

        if mask_paths is not None:
            label_mask = read_volume(mask_paths[i])[..., 0]
            label_mask = SegmentationDataset.seg_mask_to_color(
                label_mask, cls_to_color)
            if resize_to_show is not None:
                label_mask = resize_image(label_mask, resize_to_show)
            images_to_show.append(label_mask)
            titles.append('Label')
        
        key = show_images_cv2(images_to_show, titles, destroy_windows=False)
        if key == 27:
            break


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'samples_path', type=Path,
        help='Path to input image or directory with images.'
    )
    parser.add_argument(
        'checkpoint_path', type=Path,
        help='Path to model checkpoint.'
    )
    parser.add_argument(
        'config_path', type=Path,
        help='Path to model config file.'
    ),
    parser.add_argument(
        '--masks_path', type=Path, default=None,
        help='Path to label mask or directory with masks.'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run inference on (cuda/cpu).'
    ),
    parser.add_argument(
        '--resize_to_show', type=natural_int, default=None, nargs=2,
        help='Resize images to show (height, width).'
    ),
    args = parser.parse_args()

    # Validate paths
    if not args.samples_path.exists():
        raise FileNotFoundError(
            f"Samples path {str(args.samples_path)} does not exist.")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path {str(args.checkpoint_path)} does not exist.")
    if not args.config_path.exists():
        raise FileNotFoundError(
            f"Config file {str(args.config_path)} does not exist.")
    if args.masks_path is not None and not args.masks_path.exists():
        raise FileNotFoundError(
            f"Masks path {str(args.masks_path)} does not exist.")

    return args


if __name__ == '__main__':
    args = parse_args()
    main(
        args.samples_path,
        args.checkpoint_path,
        args.config_path,
        args.device,
        args.masks_path,
        args.resize_to_show
    )
