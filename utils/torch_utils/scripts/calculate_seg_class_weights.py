"""
Script to calculate segmentation class weights based on class frequencies.
"""


from pathlib import Path
import argparse
import sys

import torch
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.functions import IMAGE_EXTENSIONS, collect_paths
from utils.torch_utils.functions import read_segmentation_mask
from utils.argparse_utils import natural_int


def main(masks_dir: Path, n_classes: int, quiet: bool):

    pths = sorted(collect_paths(masks_dir, IMAGE_EXTENSIONS + ['npy']))

    class_counts = torch.zeros(n_classes, dtype=torch.float64)
    for pth in tqdm(pths, desc='Calculate class weights', disable=quiet):

        mask = (torch.from_numpy(
            read_segmentation_mask(pth, one_hot=True, n_classes=n_classes)
        ).to(dtype=torch.float64))
        class_counts += mask.sum(axis=(0, 1))

    total_samples = class_counts.sum()

    # Inverse frequency
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    logger.info(f'Inverse frequency weights: {weights.tolist()}')

    # Inverse square root frequency
    weights = 1.0 / torch.sqrt(class_counts)
    weights = weights / weights.sum()
    logger.info(f'Inverse square root frequency weights: {weights.tolist()}')

    # Balanced weights (sklearn style)
    weights = total_samples / (n_classes * class_counts)
    logger.info(f'Balanced weights: {weights.tolist()}')


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'masks_dir', type=Path, help="A dataset's masks directory.")
    parser.add_argument(
        'n_classes', type=natural_int, help='Number of classes.')
    parser.add_argument(
        '--quiet', action='store_true', help='Disable progress bar.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(masks_dir=args.masks_dir, n_classes=args.n_classes,
         quiet=args.quiet)
