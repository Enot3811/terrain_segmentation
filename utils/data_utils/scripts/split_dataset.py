"""Split images set from given dir into several subsets."""


import argparse
from pathlib import Path
from typing import List
import shutil
import random
from math import ceil


def main(
    samples_dirs: List[Path], save_dir: Path, proportions: List[float],
    random_seed: int
):
    # Collect paths and concatenate paths from images/labels dirs
    all_paths = []
    for samples_dir in samples_dirs:
        all_paths.append(sorted(samples_dir.glob('*')))

    if not all(map(lambda x: len(x) == len(all_paths[0]), all_paths)):
        raise ValueError('All dirs must contain the same number of samples.')

    all_paths = list(zip(*all_paths))

    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(all_paths)

    st_idx = 0
    for i, proportion in enumerate(proportions):
        n_samples = ceil(len(all_paths) * proportion)
        subset_pths = all_paths[st_idx:st_idx + n_samples]
        st_idx += n_samples
        subset_name = f'{save_dir.name}_{i}'
        subset_dir = save_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        subset_pths = list(zip(*subset_pths))

        for paths in subset_pths:
            dst_dir = subset_dir / paths[0].parent.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src_pth in paths:
                dst_pth = dst_dir / src_pth.name
                shutil.copy2(src_pth, dst_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--samples_dirs', type=Path, nargs='+',
        help='Paths to images/labels dirs to split.')
    parser.add_argument(
        '--save_dir', type=Path,
        help='A path to save the split images subdirectories.')
    parser.add_argument(
        '--proportions', type=float, nargs='+',
        help='Float proportions for split.')
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='A random seed for split.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(samples_dirs=args.samples_dirs,
         save_dir=args.save_dir,
         proportions=args.proportions,
         random_seed=args.random_seed)
