"""Convert class masks to one-hot encoded segmentation masks."""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.functions import collect_paths, read_image
from utils.argparse_utils import natural_int


def convert_class_mask_to_one_hot(
    input_dir: Path,
    output_dir: Path,
    n_classes: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(collect_paths(input_dir, ['png', 'jpg']))

    for image_file in tqdm(image_files, desc="Converting masks"):
        class_mask = read_image(image_file, bgr_to_rgb=False)[..., 0]

        one_hot = np.zeros((*class_mask.shape[:2], n_classes), dtype=np.uint8)
        for cls in range(n_classes):
            one_hot[..., cls] = class_mask == cls

        np.save(output_dir / image_file.with_suffix('.npy').name, one_hot)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir', type=Path,
        help='Path to the directory containing class mask images.'
    )
    parser.add_argument(
        'output_dir', type=Path,
        help=('Path to the directory to save '
              'one-hot encoded segmentation masks.')
    )
    parser.add_argument(
        'n_classes', type=natural_int,
        help='Number of classes.'
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f'Given "{args.input_dir}" directory does not exist.'
        )
    return args


def main(input_dir: Path, output_dir: Path, n_classes: int) -> None:
    """Run the main conversion process."""
    convert_class_mask_to_one_hot(input_dir, output_dir, n_classes)
    print("Conversion complete!")


if __name__ == '__main__':
    args = parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         n_classes=args.n_classes)
