"""Script to convert segmentation masks to color masks.

This script converts segmentation masks (where each pixel value represents
a class ID) to color masks (where each pixel is colored according to class
colors defined in classes.json).
"""

import json
from pathlib import Path
import argparse
from typing import Optional
import shutil
import sys

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.functions import (
    collect_paths, read_volume, save_image, IMAGE_EXTENSIONS)
from utils.torch_utils.functions import convert_seg_mask_to_color


def convert_masks(
    masks_dir: Path,
    output_dir: Optional[Path] = None,
    classes_json: Optional[Path] = None,
    quiet: bool = False
) -> None:
    """Convert segmentation masks to color masks.

    Parameters
    ----------
    masks_dir : Path
        Path to directory with segmentation masks.
    output_dir : Path
        Path to save color masks.
    classes_json : Path
        Path to classes.json file.
    quiet : bool, optional
        Whether to disable progress bar.
    """
    # Validate paths
    if not masks_dir.exists():
        raise FileNotFoundError(
            f'Masks directory does not exist: {masks_dir}'
        )
    if not masks_dir.is_dir():
        raise NotADirectoryError(
            f'Masks path is not a directory: {masks_dir}'
        )
    
    # Create output directory
    if output_dir is None:
        output_dir = masks_dir.parent / 'color_masks'
    if output_dir.exists():
        input(f'Output directory "{str(output_dir)}" already exists. '
              'Press Enter to delete it and continue...')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read classes.json
    if classes_json is None:
        classes_json = masks_dir.parent / 'classes.json'
    if not classes_json.exists():
        raise FileNotFoundError(
            f'Classes mapping file {str(classes_json)} not found.'
        )
    with open(classes_json, 'r') as f:
        classes_info = json.load(f)

    # Create id to color mapping
    class_to_id = classes_info['class_to_id']
    class_to_color = classes_info['class_to_color']
    id_to_color = {cls_id: class_to_color[cls_name]
                   for cls_name, cls_id in class_to_id.items()}

    # Collect mask paths
    mask_paths = sorted(
        collect_paths(masks_dir, IMAGE_EXTENSIONS + ['npy', 'NPY']),
        key=lambda x: x.stem
    )

    # Process each mask
    for mask_path in tqdm(
        mask_paths,
        desc='Converting masks',
        disable=quiet
    ):
        # Read mask
        seg_mask = read_volume(mask_path, bgr_to_rgb=False)
        if seg_mask.ndim == 3:
            seg_mask = seg_mask[..., 0]  # Take first channel if multi-channel

        # Convert to color
        color_mask = convert_seg_mask_to_color(seg_mask, id_to_color)

        # Save color mask
        output_path = output_dir / f'{mask_path.stem}.jpg'
        save_image(color_mask, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
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
        '--output_dir',
        type=Path,
        help='Path to save color masks. If not provided, will save in '
             'parent directory of masks_dir.'
    )
    parser.add_argument(
        '--classes_json',
        type=Path,
        help='Path to classes.json file. If not provided, will look in parent '
             'directory of masks_dir.'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable progress bar.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    convert_masks(
        args.masks_dir,
        args.output_dir,
        args.classes_json,
        args.quiet
    )
