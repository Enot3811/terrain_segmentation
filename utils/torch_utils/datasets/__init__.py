"""The module contains torch datasets classes."""

from utils.torch_utils.datasets.cvat_detection_dataset import (  # noqa
    CvatDetectionDataset)
from utils.torch_utils.datasets.abstract_torch_dataset import (  # noqa
    AbstractTorchDataset)
from utils.torch_utils.datasets.abstract_classification_dataset import (  # noqa
    AbstractClassificationDataset)
from utils.torch_utils.datasets.yolo_detection_dataset import (  # noqa
    YOLODetectionDataset)
from utils.torch_utils.datasets.segmentation_dataset import (  # noqa
    SegmentationDataset)
