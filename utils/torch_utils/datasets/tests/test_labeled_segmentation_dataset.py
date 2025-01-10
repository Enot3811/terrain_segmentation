"""Check SegmentationDataset. Iterate over it and show images with masks.

Press esc to end iteration. Press any other key to continue to the next sample.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
from cv2 import destroyAllWindows
from tqdm import tqdm
import albumentations as A

sys.path.append(str(Path(__file__).parents[4]))
from utils.data_utils.functions import show_images_cv2
from utils.argparse_utils import natural_int
from utils.torch_utils.datasets.labeled_segmentation_dataset import (
    LabeledSegmentationDataset)


def main(
    dataset_path: Path,
    crop_size: Optional[Tuple[int, int]] = None,
    batch_size: int = 4,
    random_seed: int = 42,
    one_hot_encoding: bool = False,
    check_classes: bool = False
):
    # Configure random
    torch.random.manual_seed(random_seed)

    # Prepare augmentations if needed
    if crop_size:
        transforms = A.Compose([A.RandomCrop(*crop_size)])
    else:
        transforms = None

    # Create dataset
    dataset = LabeledSegmentationDataset(
        dataset_path, transforms=transforms, one_hot_encoding=one_hot_encoding
    )

    # Check classes
    if check_classes:
        classes = dataset.collect_classes_from_masks(verbose=True)
        if len(classes) != dataset.n_classes:
            raise ValueError(
                f'Number of classes in the dataset is {len(classes)} '
                f'but {dataset.n_classes} are expected.')
        print(f'Classes in the dataset: {classes}')

    # Check loading raw dataset
    for idx in tqdm(range(len(dataset)), 'Iterating over source samples'):
        sample = dataset.get_source_sample(idx)
        image = sample['image']
        mask = sample['mask']

        if len(mask.shape) == 3:
            mask = mask.argmax(dim=0)
        elif len(mask.shape) != 2:
            raise ValueError(f'Mask shape is {mask.shape}. Expected 2 or 3.')

        # Convert mask to color
        color_mask = LabeledSegmentationDataset.seg_mask_to_color(
            mask, dataset.id_to_color)

        # Show
        key = show_images_cv2(
            [image, color_mask], ['image', 'mask'], destroy_windows=False)
        if key == 13:  # enter
            break
        if key == 27:  # esc
            destroyAllWindows()
            return
    destroyAllWindows()

    print("\nIterating over DataLoader:")
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=LabeledSegmentationDataset.collate_func)

    for batch in dataloader:
        images, masks, image_paths, mask_paths, shapes, labels = batch

        for i in range(images.shape[0]):
            # Convert tensor to numpy array
            image = images[i].permute(1, 2, 0).numpy()
            mask = masks[i].numpy()

            # Denormalize image
            image = (image * 255).astype(np.uint8)
            images_to_show = [image]
            titles = ['image']

            # Show each mask separately
            if one_hot_encoding:
                for j in range(mask.shape[0]):
                    mask_j = mask[j].astype(np.uint8)
                    mask_j = 255 * mask_j
                    images_to_show.append(mask_j)
                    titles.append(f'mask {j}')
            # Convert mask to color
            else:
                color_mask = LabeledSegmentationDataset.seg_mask_to_color(
                    mask, dataset.id_to_color)
                images_to_show.append(color_mask)
                titles.append('mask')

            # Convert labels ids to names
            sample_labels = [dataset.id_to_class[label] for label in labels[i]]

            # Show
            print(image_paths[i].name, mask_paths[i].name, sample_labels)
            key = show_images_cv2(
                images_to_show, titles, destroy_windows=False)
            if key == 27:  # esc
                destroyAllWindows()
                return
    destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('dataset_path', type=Path,
                        help='Path to the dataset directory.')
    parser.add_argument('--one_hot_encoding', action='store_true',
                        help='Use one-hot encoding for masks.')
    parser.add_argument('--crop_size', type=natural_int, nargs=2, default=None,
                        help=('Size for image and mask cropping. '
                              'If not given then cropping is not performed.'))
    parser.add_argument('--batch_size', type=natural_int, default=4,
                        help='Batch size for DataLoader.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--check_classes', action='store_true',
                        help='Check classes in the dataset.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(dataset_path=args.dataset_path, crop_size=args.crop_size,
         batch_size=args.batch_size, random_seed=args.random_seed,
         one_hot_encoding=args.one_hot_encoding,
         check_classes=args.check_classes)
