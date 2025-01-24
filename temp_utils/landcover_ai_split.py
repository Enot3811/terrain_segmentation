"""Split Landcover.AI dataset according to the given split.

Source dataset should be preprocessed by given cutter script and saved in
"split" dir.
"""

import argparse
from pathlib import Path
from typing import Optional
import shutil

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_dir', type=Path,
        help='Path to the directory containing dataset. '
             'It should contain train.txt, val.txt, and test.txt and "split" '
             'dir that contains the cut samples.'
    )
    parser.add_argument(
        '--output_dir', type=Path, default=None,
        help='Path to save split dataset. If not provided, the split dataset'
             ' will be saved in the same directory as the original dataset.'
    )
    parser.add_argument(
        '--copy', action='store_true',
        help='If true, the samples will be copied instead of moved.'
    )
    args = parser.parse_args([
        '../data/satellite/segmentation/landcover_ai_v1/',
        '--copy'
    ])

    return args


def main(dataset_dir: Path, output_dir: Optional[Path], copy: bool):
    # Check dirs
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f'Given "{dataset_dir}" directory does not exist.'
        )
    images_dir = dataset_dir / 'split'
    if not images_dir.exists():
        raise FileNotFoundError(
            f'Did not find "split" dir in "{dataset_dir}" directory.'
        )
    if output_dir is None:
        output_dir = dataset_dir

    # Iterate along splits
    for split_name in ['train', 'val', 'test']:
        # Check existence of txt file
        txt = dataset_dir / f'{split_name}.txt'
        if not txt.exists():
            print(f'Skipping "{split_name}" split as it does not exist.')
            continue

        # Prepare output dir
        split_dir = output_dir / split_name
        if split_dir.exists():
            input(f'Given "{split_dir}" directory already exists. '
                  'Press Enter to delete it and continue...')
            shutil.rmtree(split_dir)
        img_out_dir = split_dir / 'images'
        mask_out_dir = split_dir / 'masks'
        img_out_dir.mkdir(parents=True)
        mask_out_dir.mkdir(parents=True)

        # Read txt file
        with open(txt, 'r') as f:
            sample_names = f.read().splitlines()

        # Iterate along samples
        for sample_name in tqdm(sample_names,
                                desc=f'Processing "{split_name}" split'):
            sample_img = sample_name + '.jpg'
            sample_mask = sample_name + '_m.png'

            # Copy sample
            if copy:
                shutil.copy(images_dir / sample_img, img_out_dir / sample_img)
                shutil.copy(images_dir / sample_mask,
                            mask_out_dir / sample_mask)
            else:
                shutil.move(images_dir / sample_img, img_out_dir / sample_img)
                shutil.move(images_dir / sample_mask,
                            mask_out_dir / sample_mask)


if __name__ == '__main__':
    args = parse_args()
    main(args.dataset_dir, args.output_dir, args.copy)
