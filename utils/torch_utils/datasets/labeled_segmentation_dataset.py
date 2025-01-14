"""Segmentation dataset with additional labels describing classes in the mask.

It is the same dataset as `SegmentationDataset`, but with additional labels
that indicate occurrence of classes in a sample's mask.

This dataset must have additional "labels" directory with json files.
Each json file contains key "labels" with list of class ids that are present
in the corresponding sample's mask.

Any label not present in the classes.json file will be ignored.

So sample of this dataset represents a pair of image and mask and additional
metadata as image path, mask path, source image shape and labels list.

Dataset consists of
1) "images" directory that contain image or numpy files.
2) "masks" directory that contain mask image or numpy files.
3) "classes.json" file that contains class to id and class to color mappings.
4) "labels" directory that contain json files with labels.
"""


from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import json

from numpy.typing import NDArray
from torch import FloatTensor, LongTensor

from ...data_utils.functions import collect_paths
from .segmentation_dataset import SegmentationDataset


class LabeledSegmentationDataset(SegmentationDataset):
    """
    Segmentation dataset with additional labels describing classes in the mask.

    It is the same dataset as `SegmentationDataset`, but with additional labels
    that indicate occurrence of classes in a sample's mask.

    This dataset must have additional "labels" directory with json files.
    Each json file contains key "labels" with list of class ids
    that are present in the corresponding sample's mask.

    Any label not present in the classes.json file will be ignored.

    So sample of this dataset represents a pair of image and mask
    and additional metadata as image path, mask path, source image shape
    and labels list.

    Dataset consists of
    1) "images" directory that contain image or numpy files.
    2) "masks" directory that contain mask image or numpy files.
    3) "classes.json" file that contains class to id and class
       to color mappings.
    4) "labels" directory that contain json files with labels.
    """

    def __init__(
        self,
        dset_pth: Path,
        transforms: Optional[callable] = None,
        one_hot_encoding: bool = False,
    ) -> None:
        """Initialize segmentation dataset.

        Parameters
        ----------
        dset_pth : Path
            Path to the dataset directory. It expected to contain
            "images" and "masks" directories and "classes.json" file.
        transforms : Optional[callable], optional
            Transforms to apply to the samples.
            It assumed to use albumentations.
            By default None.
        one_hot_encoding : bool, optional
            Whether to convert mask to one-hot encoding, by default `True`.
        """
        super().__init__(dset_pth, transforms, one_hot_encoding)

        # Load additional labels
        self.labels_dir = dset_pth / 'labels'
        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f'Labels directory {self.labels_dir} not found.')
        self._collect_labels()

    def _collect_labels(self) -> None:
        """Collect occurrence labels from the dataset."""
        labels_paths = sorted(
            collect_paths(self.labels_dir,
                          file_extensions=['json']),
            key=lambda x: x.stem
        )
        if len(labels_paths) != len(self.samples):
            raise ValueError('Number of labels and samples are not equal.')
        
        for i, label_path in enumerate(labels_paths):
            # Read labels
            with open(label_path, 'r') as f:
                labels = json.load(f)['labels']
            # Filter with mapping
            labels = list(filter(lambda x: x in self.id_to_class, labels))
            # Add to samples
            self.samples[i]['labels'] = labels

    def get_source_sample(
        self, index: int
    ) -> Dict[str, Union[Path, NDArray, Tuple[int, int, int], List[int]]]:
        """Get original sample from the dataset.

        Return arrays without applying any transforms and tensor conversion.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        Dict[str, Union[Path, NDArray, Tuple[int, int, int], List[int]]]
            Sample dictionary with image array, mask array, image path,
            mask path, and source image shape.
        """
        sample = super().get_source_sample(index)
        sample['labels'] = self.samples[index]['labels']
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[
        FloatTensor,
        LongTensor,
        Path,
        Path,
        Tuple[int, int, int],
        List[int]
    ]:
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
        return super().postprocess_sample(sample) + (sample['labels'],)
