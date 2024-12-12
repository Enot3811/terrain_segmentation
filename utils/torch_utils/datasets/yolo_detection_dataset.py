"""Module contains torch object detection dataset in YOLO format.

YOLO format includes a yaml file that contains:

1) path: absolute path to dataset root dir
2) train: train images path relative to dataset root dir
3) val: val images path relative to dataset root dir
4) test: test images path relative to dataset root dir (optional)
5) names: class indexes with corresponding names
    for example:
    0: pedestrian
    1: people
    2: bicycle

Annotations are represented as txt files with the same names as corresponding
images files each line of which is annotation of single object. Each line has
format:
<object-class> <x_center> <y_center> <width> <height>
where:
    <object-class> is an index of class
    <x_center> is a float number in 0...1 interval that represents center of
    object's bbox by x-axis
    <y_center> is a float number in 0...1 interval that represents center of
    object's bbox by y-axis
    <width> is a float number in 0...1 interval that represents width of
    object's bbox
    <height> is a float number in 0...1 interval that represents height of
    object's bbox
Annotations are stored in `labels/{subset}` subdirectory.
If image has no objects, then .txt file is not required.

Example of dataset structure:
dataset_root_dir
├── {dset_pth.yaml}
├── images
|   ├── train
|   |   ├── {train_sample_name_0}.jpg
|   |   ├── {train_sample_name_1}.jpg
|   |   ├── {train_sample_name_2}.jpg
|   |   ├── ...
|   |   ├── {train_sample_name_n-1}.jpg
|   |   ├── {train_sample_name_n}.jpg
|   ├── val
|       ├── {val_sample_name_0}.jpg
|       ├── {val_sample_name_1}.jpg
|       ├── {val_sample_name_2}.jpg
|       ├── ...
|       ├── {val_sample_name_n-1}.jpg
|       ├── {val_sample_name_n}.jpg
├── labels
    ├── train
    |   ├── {train_sample_name_0}.txt
    |   ├── {train_sample_name_1}.txt
    |   ├── {train_sample_name_2}.txt
    |   ├── ...
    |   ├── {train_sample_name_n-1}.txt
    |   ├── {train_sample_name_n}.txt
    ├── val
        ├── {val_sample_name_0}.jpg
        ├── {val_sample_name_1}.jpg
        ├── {val_sample_name_2}.jpg
        ├── ...
        ├── {val_sample_name_n-1}.jpg
        ├── {val_sample_name_n}.jpg
"""

from pathlib import Path
from typing import Union, Any, List, Tuple, Dict, Callable, Optional
import yaml
from PIL import Image

import torch
from torch import FloatTensor, LongTensor

from .abstract_detection_dataset import AbstractDetectionDataset
from ..functions import image_numpy_to_tensor
from ...data_utils.functions import (
    read_volume, collect_paths, IMAGE_EXTENSIONS)


class YOLODetectionDataset(AbstractDetectionDataset):
    """YOLO compatible object detection dataset."""

    def __init__(
        self,
        dset_pth: Union[Path, str],
        transforms: Callable = None,
        subset: str = 'train'
    ) -> None:
        """Initialize dataset.
        
        Initialization contains:
        - loading and checking given yaml file and saving paths to images
        and labels dirs of given subset
        - samples collection
        - class labels collection and checking
        - transforms assignment

        Parameters
        ----------
        dset_pth : Union[Path, str]
            Path to datasets's yaml file.
        transforms : Callable, optional
            Transforms that performs on sample.
            Required that it has `albumentations.Compose` like structure.
            By default is `None`.
        class_to_index : Dict[str, int], optional
            User-defined class to index mapping. It required that this dict
            contains all classes represented in the dataset.
            By default is `None`.
        """
        super().__init__(dset_pth=dset_pth, transforms=transforms,
                         subset=subset)

    def _parse_dataset_pth(
        self,
        dset_pth: Union[Path, str]
    ) -> Path:
        """Check passed YOLO yaml and parse included paths.

        Return yaml path as a dataset path and initialize `self.image_dir`,
        `self.labels_dir` and `self.root_dir`.

        Parameters
        ----------
        dset_pth : Union[Path, str]
            Path to dataset's yaml file.

        Returns
        -------
        Path
            Dataset's root path.

        Raises
        ------
        ValueError
            Raise when `dset_pth` has wrong type.
        FileNotFoundError
            Raise when `dset_pth` does not exists.
        """
        dset_pth = super()._parse_dataset_pth(dset_pth)
        with open(dset_pth, 'r') as f:
            dataset_config = yaml.safe_load(f)
        if self.subset not in dataset_config:
            raise ValueError(f'Dataset has no {self.subset} subset.')
        self.root_dir = Path(dataset_config['path'])
        self.image_dir = self.root_dir / dataset_config[self.subset]
        self.labels_dir = self.root_dir / 'labels' / self.subset
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f'Image directory not found: "{str(self.image_dir)}"')
        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f'Labels directory not found: "{str(self.labels_dir)}"')
        return dset_pth

    def _collect_samples(
        self, dset_pth: Path
    ) -> List[Dict[str, Any]]:
        """Iterate over annotations files and collect samples.

        Each sample is a `dict` that contain: "img_pth", "labels", "bboxes"
        and "shape". Where "labels" is a `list` of class indexes and "bboxes"
        is a `list` of `FloatBbox` objects in normalized "cxcywh" format.

        Parameters
        ----------
        dset_pth : Path
            Is not used. Instead of it `self.image_dir` and
            `self.labels_dir` are used.

        Returns
        -------
        List[Dict[str, Any]]
            Collected samples from the dataset.
        """
        image_paths = sorted(collect_paths(self.image_dir, IMAGE_EXTENSIONS))
        label_paths = sorted(collect_paths(self.labels_dir, ['txt']))
        if len(image_paths) != len(label_paths):
            raise ValueError(
                f'Number of images and labels are different: '
                f'num_images={len(image_paths)}, '
                f'num_labels={len(label_paths)}'
            )
        samples = []
        for img_pth, label_pth in list(zip(image_paths, label_paths)):

            with open(label_pth, 'r') as f:
                annot_lines = f.readlines()
            img = Image.open(img_pth)
            shape = img.size
            img_bboxes = []
            img_labels = []
            for annot_line in annot_lines:
                split_annot = annot_line.split()
                class_idx = int(split_annot[0])
                x_center = float(split_annot[1])
                y_center = float(split_annot[2])
                width = float(split_annot[3])
                height = float(split_annot[4])

                # Check correctness of bbox
                x1, y1, x2, y2 = self.convert_bboxes_to_xyxy(
                    [[x_center, y_center, width, height]], shape)[0]
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, shape[1])
                y2 = min(y2, shape[0])
                if x1 >= x2 or y1 >= y2:
                    print(f'Invalid bbox: {x1}, {y1}, {x2}, {y2} '
                          f'in "{str(img_pth)}" sample.')
                    continue
                x_center = (x1 + x2) / 2 / shape[1]
                y_center = (y1 + y2) / 2 / shape[0]
                width = (x2 - x1) / shape[1]
                height = (y2 - y1) / shape[0]

                img_bboxes.append((x_center, y_center, width, height))
                img_labels.append(class_idx)
            samples.append({
                'img_pth': img_pth,
                'bboxes': img_bboxes,
                'labels': img_labels,
                'shape': shape
            })
        return samples

    def _collect_class_labels(
        self, samples: List[Dict[str, Any]],
        class_to_index: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Read class labels from yaml file.

        The verification of samples' labels will be performed in
        `self._collect_samples` method.

        Parameters
        ----------
        samples : List[Dict[str, Any]]
            Dataset samples.
        class_to_index : Dict[str, int], optional
            Is not used. Instead of it dataset's yaml file is used.
        
        Returns
        -------
        Tuple[Dict[str, int], Dict[int, str]]
            Class labels mapping.
        """
        # Collect labels from samples
        labels = set()
        for sample in samples:
            labels.update(sample['labels'])
        # Get labels from yaml file
        with open(self.dset_pth, 'r') as f:
            dataset_config = yaml.safe_load(f)
        id_to_cls = dataset_config['names']
        if isinstance(id_to_cls, list):
            id_to_cls = {cls: i for i, cls in enumerate(id_to_cls)}
        # Check that all labels are presented in the dataset
        if not labels.issubset(id_to_cls.keys()):
            raise ValueError(
                "Found labels that are not presented "
                "in the dataset's yaml file.")
        cls_to_id = {v: k for k, v in id_to_cls.items()}
        return cls_to_id, id_to_cls
    
    def get_source_sample(
        self, index: int
    ) -> Dict[str, Any]:
        """Get YOLO object detection sample.

        Sample represented as a dict that contains "image" `ndarray`,
        "bboxes" `list[list[float]]`, "labels" `list[int]`, "img_pth" `Path`
        and "shape" `Tuple[int, int]`.
        Bounding boxes is in normalized "cxcywh" format.

        Parameters
        ----------
        index : int
            Index of sample.

        Returns
        -------
        Dict[str, Any]
            YOLO object detection sample by index.
        """
        sample_annots = self.get_samples_annotations()[index]
        img_pth = sample_annots['img_pth']
        labels = sample_annots['labels']
        bboxes = sample_annots['bboxes']
        shape = sample_annots['shape']
        image = read_volume(img_pth)

        sample = {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'img_pth': img_pth,
            'shape': shape
        }
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[FloatTensor, FloatTensor, LongTensor, Path, Tuple[int, int]]:
        """Convert YOLO object detection sample to tensors and pack to tuple.

        Image `uint8 ndarray` will be converted to `FloatTensor` in 0...1
        interval. Bboxes to FloatTensor and classes to IntTensor.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample in original format.

        Returns
        -------
        Tuple[FloatTensor, FloatTensor, LongTensor, Path, Tuple[int, int]]
            Normalized image, bboxes in yolo format, labels, image path,
            and source image `(height, width)`.
        """
        # Convert image, bboxes and classes to tensor
        sample['image'] = image_numpy_to_tensor(
            sample['image'], dtype=torch.float32) / 255
        sample['bboxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
        sample['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)
        return (sample['image'], sample['bboxes'], sample['labels'],
                sample['img_pth'], sample['shape'])
    
    def apply_transforms(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply passed transforms on the sample.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample to transform.

        Returns
        -------
        Dict[str, Any]
            Transformed sample.
        """
        # Using albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=sample['image'],
                                          bboxes=sample['bboxes'],
                                          classes=sample['labels'])
            sample['image'] = transformed['image']  # ArrayLike
            sample['bboxes'] = transformed['bboxes']  # list[list[float]]
            sample['labels'] = transformed['classes']  # list[int]
        return sample
    
    @staticmethod
    def collate_func(
        batch: List[Tuple[
            FloatTensor,
            FloatTensor,
            LongTensor,
            Path,
            Tuple[int, int]
        ]]
    ) -> Tuple[
        FloatTensor,
        List[FloatTensor],
        List[LongTensor],
        List[Path],
        List[Tuple[int, int]]
    ]:
        """Collate function for the dataset.

        Parameters
        ----------
        batch : List[Tuple[
            FloatTensor,
            FloatTensor,
            LongTensor,
            Path,
            Tuple[int, int]
        ]]
            List of samples.

        Returns
        -------
        Tuple[
            FloatTensor,
            FloatTensor,
            LongTensor,
            List[Path],
            List[Tuple[int, int]]
        ]
            Batch of samples.
        """
        images, bboxes, labels, image_paths, shapes = zip(*batch)
        
        # Concatenate images along a new axis (batch dimension)
        images = torch.stack(images, dim=0)
        # Pack bboxes, labels, image paths and shapes into lists
        bboxes = list(bboxes)
        labels = list(labels)
        image_paths = list(image_paths)
        shapes = list(shapes)
        
        return images, bboxes, labels, image_paths, shapes
    
    @staticmethod
    def convert_bboxes_to_xyxy(
        bboxes: List[List[float]],
        img_shape: Tuple[int, int]
    ) -> List[List[float]]:
        """Convert bboxes from normalized "cxcywh" format to "xyxy" format.

        Parameters
        ----------
        bboxes : List[List[float]]
            Bounding boxes in normalized "cxcywh" format.
        img_shape : Tuple[int, int]
            Image shape.

        Returns
        -------
        List[List[float]]
            Bounding boxes in normalized "xyxy" format.
        """
        img_h, img_w = img_shape
        bboxes_xyxy = []
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h
            bboxes_xyxy.append([x1, y1, x2, y2])
        return bboxes_xyxy
