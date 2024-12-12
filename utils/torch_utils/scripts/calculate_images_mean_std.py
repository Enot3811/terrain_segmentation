"""Script to calculate image dataset's channel wise std and mean values."""

from pathlib import Path
import argparse
import sys

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.functions import (
    read_image, IMAGE_EXTENSIONS, collect_paths)


def main(image_dir: Path, show_progress: bool, device: str):

    if device == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(device)

    pths = sorted(collect_paths(image_dir, IMAGE_EXTENSIONS))

    means = torch.empty((len(pths), 3), dtype=torch.float32, device=device)
    stds = torch.empty((len(pths), 3), dtype=torch.float32, device=device)
    for i, pth in tqdm(enumerate(pths), 'Calculate mean and std',
                       disable=not show_progress):
        img = (torch.from_numpy(read_image(pth))
               .to(dtype=torch.float32, device=device) / 255.0)
        means[i] = torch.mean(img, axis=(0, 1))
        stds[i] = torch.std(img, axis=(0, 1))
    mean = torch.mean(means, axis=0)
    std = torch.mean(stds, axis=0)

    print('Mean:', mean.tolist())
    print('Std:', std.tolist())


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'image_dir', type=Path, help="A dataset's image directory.")
    parser.add_argument(
        '--show_progress', action='store_true', help='Show progress bar.')
    parser.add_argument(
        '--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
        help=('The device on which calculations are performed. '
              '"auto" selects "cuda" when it is possible.'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(image_dir=args.image_dir, show_progress=args.show_progress,
         device=args.device)
