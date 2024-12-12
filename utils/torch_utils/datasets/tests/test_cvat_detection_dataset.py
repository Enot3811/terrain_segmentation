"""Check `CvatDetectionDataset`. Iterate over it and show images with bboxes.

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
from utils.torch_utils.functions import draw_bounding_boxes
from utils.argparse_utils import natural_int
from utils.torch_utils.datasets.cvat_detection_dataset import (
    CvatDetectionDataset)


def main(
    dataset_path: Path,
    crop_size: Optional[Tuple[int, int]] = None,
    batch_size: int = 4,
    random_seed: int = 42,
):
    # Configure random
    torch.random.manual_seed(random_seed)

    # Prepare augmentations if needed
    if crop_size:
        transforms = A.Compose(
            [A.RandomCrop(*crop_size)],
            bbox_params=A.BboxParams(
                format='pascal_voc', label_fields=['classes']))
    else:
        transforms = None

    # Create dataset
    dataset = CvatDetectionDataset(
        dataset_path, transforms=transforms
    )

    # Check classes
    print(f'Classes in the dataset: {dataset.get_class_to_index()}')

    for idx in tqdm(range(len(dataset)), 'Iterating over source samples'):
        sample = dataset.get_source_sample(idx)
        image = sample['image']
        bboxes = sample['bboxes']
        labels = sample['labels']

        # Draw bboxes
        bbox_image = draw_bounding_boxes(
            image=image, bboxes=bboxes, class_labels=labels, line_width=1)

        # Show
        key = show_images_cv2(
            [image, bbox_image], ['image', 'bboxes'], destroy_windows=False)
        if key == 13:  # enter
            break
        if key == 27:  # esc
            destroyAllWindows()
            return
    destroyAllWindows()

    print("\nIterating over DataLoader:")
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=CvatDetectionDataset.collate_func,
                            num_workers=0)

    for batch in dataloader:
        images, batch_bboxes, batch_labels, image_paths, shapes = batch

        for i in range(images.shape[0]):
            # Convert tensor to numpy array
            image = images[i].permute(1, 2, 0).numpy()
            bboxes = batch_bboxes[i].numpy()
            labels = batch_labels[i].numpy()

            # Denormalize image
            image = (image * 255).astype(np.uint8)

            # Draw bboxes
            bbox_image = draw_bounding_boxes(
                image=image, bboxes=bboxes, class_labels=labels, line_width=1)

            # Show
            print(image_paths[i].name)
            key = show_images_cv2(
                [image, bbox_image], ['image', 'bboxes'],
                destroy_windows=False)
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
    parser.add_argument('--crop_size', type=natural_int, nargs=2, default=None,
                        help=('Size for image and mask cropping. '
                              'If not given then cropping is not performed.'))
    parser.add_argument('--batch_size', type=natural_int, default=4,
                        help='Batch size for DataLoader.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(dataset_path=args.dataset_path, crop_size=args.crop_size,
         batch_size=args.batch_size, random_seed=args.random_seed)
