"""Torch dataset for segmentation tasks.
    
Sample of this dataset represents a pair of image and mask and additional
metadata as image path, mask path, and source image shape.

Dataset consists of
1) "images" directory that contain image or numpy files.
2) "masks" directory that contain mask image or numpy files.
3) "classes.json" file that contains class to id and class to color mappings.

If masks have some classes that are not present in the classes.json file,
then they will be discarded during sample preprocessing.
If classes.json does not have class to color mapping, then colors will be
randomly assigned.

Unique classes that encountered in the dataset can be collected by calling
`collect_classes_from_masks` method.
"""


from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import json

import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from loguru import logger

from ..functions import convert_seg_mask_to_one_hot, convert_seg_mask_to_color
from ...data_utils.functions import (
    collect_paths, IMAGE_EXTENSIONS, read_volume, prepare_path,
    generate_class_to_colors)


class SegmentationDataset(Dataset):
    """Torch dataset for segmentation tasks.
    
    Sample of this dataset represents a pair of image and mask and additional
    metadata as image path, mask path, and source image shape.

    Dataset consists of
    1) "images" directory that contain image or numpy files.
    2) "masks" directory that contain mask image or numpy files.
    3) "classes.json" file that contains class to id and class to color
       mappings.

    If masks have some classes that are not present in the classes.json file,
    then they will be discarded during sample preprocessing.
    If classes.json does not have class to color mapping, then colors will be
    randomly assigned.

    Unique classes that encountered in the dataset can be collected by calling
    `collect_classes_from_masks` method.
    """

    def __init__(
        self,
        dataset_path: Path,
        transforms: Optional[callable] = None,
        one_hot_encoding: bool = False,
    ) -> None:
        """Initialize segmentation dataset.

        Parameters
        ----------
        dataset_path : Path
            Path to the dataset directory. It expected to contain
            "images" and "masks" directories and "classes.json" file.
        transforms : Optional[callable], optional
            Transforms to apply to the samples.
            It assumed to use albumentations.
            By default None.
        one_hot_encoding : bool, optional
            Whether to convert mask to one-hot encoding, by default `True`.
        """
        # Dataset path
        self.dataset_pth = prepare_path(dataset_path)
        self.image_dir = self.dataset_pth / 'images'
        self.mask_dir = self.dataset_pth / 'masks'
        self.classes_json = self.dataset_pth / 'classes.json'

        # Check paths
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f'Image directory {self.image_dir} does not exist.')
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f'Mask directory {self.mask_dir} does not exist.')
        if not self.classes_json.exists():
            raise FileNotFoundError(
                f'Classes mapping file {self.classes_json} does not exist.')

        # Load classes, create mappings
        self._load_classes_file()

        # Load samples paths
        self.samples = self._collect_samples()

        # Transforms and preprocessing
        self.transforms = transforms
        self.one_hot_encoding = one_hot_encoding

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(
        self, index: int
    ) -> Tuple[FloatTensor, LongTensor, Path, Path, Tuple[int, int, int]]:
        sample = self.get_source_sample(index)
        sample = self.preprocess_sample(sample)
        sample = self.apply_transforms(sample)
        sample = self.postprocess_sample(sample)
        return sample

    def get_source_sample(
        self, index: int
    ) -> Dict[str, Union[Path, NDArray, Tuple[int, int, int]]]:
        """Get original sample from the dataset.

        Return arrays without applying any transforms and tensor conversion.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        Dict[str, Union[Path, NDArray, Tuple[int, int, int]]]
            Sample dictionary with image array, mask array, image path,
            mask path, and source image shape.
        """
        sample = self.samples[index]
        image = read_volume(sample['image_path'])
        mask = read_volume(sample['mask_path'], bgr_to_rgb=False)
        if mask.ndim == 3:
            mask = mask[..., 0]  # Take first channel if mask is image
        return {
            'image': image,
            'mask': mask,
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
            'shape': image.shape
        }
    
    def preprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process sample before applying transforms.

        Filter target masks by classes defined in `class_to_id` mapping.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary with image and mask.
        """
        # Filter target masks by classes defined in `class_to_id` mapping
        discard_mask = np.ones_like(sample['mask'], dtype=np.bool_)
        for cls_id in self.id_to_class:
            discard_mask[sample['mask'] == cls_id] = False
        sample['mask'][discard_mask] = 0
        return sample
    
    def apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to the sample.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary with image and mask.

        Returns
        -------
        Dict[str, Any]
            Sample dictionary with transformed image and mask.
        """
        if self.transforms:
            transformed = self.transforms(
                image=sample['image'], mask=sample['mask'])
            sample['image'] = transformed['image']
            sample['mask'] = transformed['mask']
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[FloatTensor, LongTensor, Path, Path, Tuple[int, int, int]]:
        """Process sample after applying transforms.

        Normalize, convert to float tensor, convert mask to long tensor.
        And convert mask to one-hot encoding if needed.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary with image and mask.

        Returns
        -------
        Tuple[FloatTensor, LongTensor, Path, Path, Tuple[int, int, int]]
            Normalized image, segmentation mask, image path, mask path,
            and source image shape.
        """
        # Convert mask
        mask = torch.tensor(sample['mask'], dtype=torch.long)  # h, w
        if self.one_hot_encoding:
            mask = self.seg_mask_to_one_hot(mask, self.n_classes)  # c, h, w

        # Convert image to float tensor and normalize
        image = torch.tensor(
            sample['image'].transpose(2, 0, 1).astype(np.float32) / 255)

        return (image, mask, sample['image_path'], sample['mask_path'],
                sample['shape'])
    
    def _load_classes_file(self) -> None:
        """Load classes file.

        Load classes info from file and create all mappings.

        Raises
        ------
        ValueError
            If class to id mapping is not found in the classes.json file.
        """
        with open(self.classes_json, 'r') as f:
            classes_info = json.load(f)

            # Class to id mapping
            if 'class_to_id' not in classes_info:
                raise ValueError(
                    'Class to id mapping is not found '
                    'in the classes.json file.')
            self.class_to_id = classes_info['class_to_id']

            self.id_to_class = {v: k for k, v in self.class_to_id.items()}
            self.n_classes = len(self.class_to_id)

            # Class to color mapping
            if 'class_to_color' not in classes_info:
                logger.warning(
                    'Class to color mapping is not found '
                    'in the classes.json file. Class colors will be '
                    'randomly assigned.')
                self.class_to_color = self.generate_class_to_colors(
                    self.n_classes)
            else:
                self.class_to_color = classes_info['class_to_color']
                self.class_to_color = {
                    k: v for k, v in self.class_to_color.items()
                    if k in self.class_to_id}
            
            self.color_to_class = {
                tuple(v): k for k, v in self.class_to_color.items()}
            
            # Id to color mapping
            self.id_to_color = {self.class_to_id[k]: tuple(v)
                                for k, v in self.class_to_color.items()}
            self.color_to_id = {v: k for k, v in self.id_to_color.items()}

    def _collect_samples(self) -> List[Dict[str, Path]]:
        """Collect samples from the dataset.

        Returns
        -------
        List[Dict[str, Path]]
            List of dictionaries with paths to images and masks.

        Raises
        ------
        ValueError
            If number of images and masks are not equal.
        """
        img_paths = sorted(
            collect_paths(self.image_dir,
                          file_extensions=IMAGE_EXTENSIONS + ['npy', 'NPY']),
            key=lambda x: x.stem
        )
        mask_paths = sorted(
            collect_paths(self.mask_dir,
                          file_extensions=IMAGE_EXTENSIONS + ['npy', 'NPY']),
            key=lambda x: x.stem
        )
        
        if len(img_paths) != len(mask_paths):
            raise ValueError('Number of images and masks are not equal.')
        
        return [{'image_path': img_path, 'mask_path': mask_path}
                for img_path, mask_path in zip(img_paths, mask_paths)]

    def collect_classes_from_masks(self, verbose: bool = False) -> Set[int]:
        """Get unique labels of pixels from masks in the dataset.

        This function can take a long time to execute due to the need
        to actually read all the masks files in the dataset.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show a progress bar, by default `False`.

        Returns
        -------
        Set[int]
            Set of unique labels in the dataset.
        """
        labels = set()
        for i in tqdm(range(len(self)), desc='Collecting labels from masks',
                      disable=not verbose):
            sample = self.get_source_sample(i)
            mask = sample['mask']
            labels.update(set(np.unique(mask)))
        return labels
    
    @staticmethod
    def seg_mask_to_one_hot(
        seg_mask: NDArray, n_classes: int, cls_dim: int = 0
    ) -> NDArray:
        """Convert class mask to one-hot encoding.

        Parameters
        ----------
        seg_mask : LongTensor
            Segmentation mask with shape `(h, w)` or `(b, h, w)` for batch.
        n_classes : int
            Number of classes in given segmentation mask.
        cls_dim : int, optional
            Dimension to put classes in one-hot mask. By default is `0`.

        Returns
        -------
        NDArray
            One-hot encoding with shape `(h, w, n_classes)`.
        """
        return convert_seg_mask_to_one_hot(
            seg_mask, n_classes, cls_dim=cls_dim)

    @staticmethod
    def one_hot_to_seg_mask(
        one_hot: NDArray
    ) -> NDArray:
        """Convert one-hot encoding to class mask.

        Parameters
        ----------
        one_hot : NDArray
            One-hot encoding with shape `(h, w, n_classes)`.

        Returns
        -------
        NDArray
            Class mask with shape `(h, w)`.
        """
        return np.argmax(one_hot, axis=-1)
    
    @staticmethod
    def seg_mask_to_color(
        seg_mask: NDArray, cls_to_color: Dict[int, Tuple[int, int, int]]
    ) -> NDArray:
        """Convert segmentation mask to color mask.

        If `seg_mask` has class ids that are not in `cls_to_color`
        then they will be colored in black `(0, 0, 0)`.

        Parameters
        ----------
        seg_mask : NDArray
            Segmentation mask 2D array.
        cls_to_colors : Dict[int, Tuple[int, int, int]]
            Mapping from class labels to colors.

        Returns
        -------
        NDArray
            Color mask 3D array.
        """
        return convert_seg_mask_to_color(seg_mask, cls_to_color)

    @staticmethod
    def color_mask_to_seg(
        color_mask: NDArray, colors_to_cls: Dict[Tuple[int, int, int], int]
    ) -> NDArray:
        """Convert color mask to segmentation mask.

        Parameters
        ----------
        color_mask : NDArray
            Color mask 3D array.
        colors_to_cls : Dict[Tuple[int, int, int], int]
            Mapping from colors to class labels.

        Returns
        -------
        NDArray
            Segmentation mask 2D array.
        """
        seg_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
        for color, cls in colors_to_cls.items():
            mask = np.all(color_mask == color, axis=-1)
            seg_mask[mask] = cls
        return seg_mask
    
    @staticmethod
    def collate_func(batch: List[Any]) -> Tuple[Any, ...]:
        """Collate function for the dataset.
    
        Automatically handles:
        - Stacking tensors (like images and masks)
        - Converting lists of other types (paths, shapes, labels, etc.)
        
        Parameters
        ----------
        batch : List[Any]
            List of samples, where each sample is a tuple of elements
        
        Returns
        -------
        Tuple[Any, ...]
            Batch of collated elements
        """
        batch = list(zip(*batch))
    
        collated = []
        for elements in batch:
            # If elements are tensors, stack them
            if torch.is_tensor(elements[0]):
                collated.append(torch.stack(elements, dim=0))
            # Otherwise, convert to list
            else:
                collated.append(list(elements))
        
        return tuple(collated)
    
    @staticmethod
    def generate_class_to_colors(
        n_classes: int,
        classes_ids: Optional[List[int]] = None
    ) -> Dict[int, Tuple[int, int, int]]:
        """Generate random colors for classes.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        classes_ids : Optional[List[int]], optional
            List of classes ids to generate colors for. If `None` then colors
            will be generated for all classes. By default `None`.

        Returns
        -------
        Dict[int, Tuple[int, int, int]]
            Mapping from class labels to colors.
        """
        return generate_class_to_colors(n_classes, classes_ids)
