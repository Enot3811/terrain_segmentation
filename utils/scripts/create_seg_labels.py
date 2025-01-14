"""Script to collect and save class labels present in segmentation masks.

This script analyzes segmentation masks in a dataset and creates JSON files
containing information about which classes are present in each mask.
The labels are saved in the 'labels' directory with the same name as the
corresponding mask file.
"""

import json
from pathlib import Path
import argparse
import shutil
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.functions import (
    collect_paths, read_volume, IMAGE_EXTENSIONS)


def collect_masks_labels(masks_dir: Path, quiet: bool) -> None:
    """Collect and save labels for all masks in the directory.

    Parameters
    ----------
    mask_dir : Path
        Path to the directory with segmentation masks.
    quiet : bool
        Whether to disable progress bar.

    Raises
    ------
    FileNotFoundError
        If masks directory doesn't exist.
    """
    if not masks_dir.exists():
        raise FileNotFoundError(
            f'Masks directory "{str(masks_dir)}" does not exist.')

    # Create labels directory
    labels_dir = masks_dir.parent / 'labels'
    if labels_dir.exists():
        input(f'Labels directory "{str(labels_dir)}" already exists. '
              'Press Enter to delete it and continue...')
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(exist_ok=True)

    # Collect mask paths
    mask_paths = sorted(
        collect_paths(
            masks_dir,
            file_extensions=IMAGE_EXTENSIONS + ['npy', 'NPY']
        ),
        key=lambda x: x.stem
    )

    # Process each mask
    for mask_path in tqdm(
        mask_paths,
        desc='Collecting labels',
        disable=quiet
    ):
        # Get labels
        mask = read_volume(mask_path, bgr_to_rgb=False)
        if mask.ndim == 3:
            mask = mask[..., 0]  # Take first channel if mask is image
        labels = np.unique(mask).tolist()

        # Save to JSON
        save_path = labels_dir / f'{mask_path.stem}.json'
        with open(save_path, 'w') as f:
            json.dump({'labels': sorted(labels)}, f)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'masks_dir',
        type=Path,
        help='Path to the directory with segmentation masks.'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable progress bar'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    collect_masks_labels(args.masks_dir, args.quiet)
