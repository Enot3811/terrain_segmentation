"""Convert YOLO dataset to CVAT format.

YOLO format is:
dataset_dir
├── data.yaml
├── train
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt
├── val (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt
├── test (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt

Where each txt corresponds to a jpg with the same name.
Txt file consists of lines like: `"cls_id cx cy h w"`, where `cls_id` is an int
that corresponds to some class and `"cxcywh"` is a bounding box of object.
Every value of bounding box is normalized relative to image shape.
"""


import argparse
from pathlib import Path
import sys
from typing import List, Tuple
import shutil
import yaml
from PIL import Image

from tqdm import tqdm
from torch import tensor
from torchvision.ops import box_convert

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.cvat_functions import create_cvat_xml_from_dataset
from utils.data_utils.functions import (
    read_volume, IMAGE_EXTENSIONS, collect_paths)


def main(yolo_yaml: Path, save_dir: Path, copy_images: bool, verbose: bool):
    
    # Check existing
    if save_dir.exists():
        input(f'Specified directory "{str(save_dir)}" already exists. '
              'If continue, this directory will be deleted. '
              'Press enter to continue.')
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    # Read yolo config, get classes and ids
    with open(yolo_yaml) as f:
        yolo_config = yaml.safe_load(f)
    yolo_pth = Path(yolo_config['path'])
    id_to_cls = yolo_config['names']
    if isinstance(id_to_cls, list):
        id_to_cls = {i: cls for i, cls in enumerate(id_to_cls)}
    cls_names = list(id_to_cls.values())

    # Iterate over possible subsets
    for subset in ['train', 'val', 'test']:
        
        # Process only existing subsets
        subset_path = yolo_config.get(subset, None)
        if subset_path is None:
            continue

        # Get source paths
        subset_img_dir = yolo_pth / subset_path
        subset_labels_dir = yolo_pth / 'labels' / subset

        # Prepare cvat paths and dirs
        cvat_dset_path = save_dir / subset
        cvat_images_dir = cvat_dset_path / 'images'
        cvat_xml = cvat_dset_path / 'annotations.xml'
        cvat_dset_path.mkdir()
        if copy_images:
            cvat_images_dir.mkdir()

        # Read yolo samples
        yolo_images = sorted(collect_paths(subset_img_dir, IMAGE_EXTENSIONS))
        dataset_annots = []
        for img_pth in tqdm(yolo_images,
                            desc='Convert dataset',
                            disable=not verbose):

            sample_name = img_pth.stem
            txt_pth = subset_labels_dir / f'{sample_name}.txt'

            # Read image's shape and copy image if needed
            if img_pth.suffix.lower() not in IMAGE_EXTENSIONS:
                sample_img = read_volume(img_pth)
                shape = sample_img.shape[:2]
            else:
                sample_img = Image.open(img_pth)
                shape = sample_img.size
            if copy_images:
                dst_pth = cvat_images_dir / img_pth.name
                shutil.copy2(img_pth, dst_pth)

            # Read sample annotations
            if not txt_pth.exists():
                sample_annots = []
            else:
                with open(txt_pth) as f:
                    sample_annots = f.readlines()

            # Convert annotations
            labels: List[str] = []
            bboxes: List[Tuple[float, float, float, float]] = []
            for sample_annot in sample_annots:
                split_annot = sample_annot.split(' ')
                label = id_to_cls[int(split_annot[0])]
                bbox = tensor(list(map(float, split_annot[1:])))
                bbox[[1, 3]] *= shape[0]
                bbox[[0, 2]] *= shape[1]
                bbox = box_convert(bbox, 'cxcywh', 'xyxy')
                bbox[bbox < 0] = 0
                if bbox[0] >= bbox[2]:
                    print(
                        f'Annotation "{sample_annot}" '
                        f'of sample "{sample_name}" has wrong bbox. '
                        'After conversion to "xyxy" format '
                        f'x_max is less than x_min "{bbox[0]} {bbox[2]}".\n'
                        'Skipping...')
                elif bbox[1] >= bbox[3]:
                    print(
                        f'Annotation "{sample_annot}" '
                        f'of sample "{sample_name}" has wrong bbox. '
                        'After conversion to "xyxy" format '
                        f'y_max is less than y_min "{bbox[1]} {bbox[3]}".\n'
                        'Skipping...')
                else:
                    bbox = bbox.tolist()
                    labels.append(label)
                    bboxes.append(bbox)

            dataset_annots.append({
                'img_pth': img_pth,
                'shape': shape,
                'bboxes': bboxes,
                'labels': labels
            })

        # Create cvat xml
        create_cvat_xml_from_dataset(
            cvat_xml, cls_names, dataset_annots, subset, verbose)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'yolo_yaml', type=Path,
        help='Path to YOLO dataset yaml file to convert.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the converted CVAT dataset.')
    parser.add_argument(
        '--copy_images', action='store_true',
        help='Whether to copy dataset images.')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to show progress of converting.')
    args = parser.parse_args()
    
    if not args.yolo_yaml.exists():
        raise FileNotFoundError(
            f'Dataset yaml file "{str(args.yolo_yaml)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(yolo_yaml=args.yolo_yaml, save_dir=args.save_dir,
         copy_images=args.copy_images, verbose=args.verbose)
