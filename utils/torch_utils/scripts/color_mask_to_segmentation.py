"""Convert color masks to segmentation masks."""

import argparse
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.functions import read_image, save_image


COLOR_TO_CLS = {
    (255, 255, 255): 0,  # Background
    (255, 0, 0): 1,      # Road
    (0, 255, 0): 2,      # Trees
    (0, 0, 255): 3,      # Water
}


def convert_color_mask_to_segmentation(
    input_dir: Path,
    output_dir: Path
) -> None:
    """Convert color masks to segmentation masks.

    Args:
        input_dir: Path to the directory containing color mask images.
        output_dir: Path to the directory to save segmentation masks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob('*.png'))

    for image_file in tqdm(image_files, desc="Converting masks"):
        color_mask_array = read_image(image_file)

        segmentation_mask = np.zeros(
            color_mask_array.shape[:2],
            dtype=np.uint8
        )

        for color, cls in COLOR_TO_CLS.items():
            mask = np.all(color_mask_array == color, axis=-1)
            segmentation_mask[mask] = cls

        save_image(segmentation_mask, output_dir / image_file.name,
                   rgb_to_bgr=False)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir', type=Path,
        help='Path to the directory containing color mask images.'
    )
    parser.add_argument(
        'output_dir', type=Path,
        help='Path to the directory to save segmentation masks.'
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f'Given "{args.input_dir}" directory does not exist.'
        )
    return args


def main(input_dir: Path, output_dir: Path) -> None:
    """Run the main conversion process."""
    convert_color_mask_to_segmentation(input_dir, output_dir)
    print("Conversion complete!")


if __name__ == '__main__':
    args = parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
