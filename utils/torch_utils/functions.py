"""Tools for working with PyTorch.

Contains utilities for:
    - Image processing
    - Segmentation masks processing
    - Training statistics tools
"""


from typing import List, Tuple, Union, Dict, Optional, Type
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from torch import Tensor, LongTensor
from torch.nn import Module
from torchvision.ops import box_convert
from tensorboard.backend.event_processing.event_file_loader import (
    RawEventFileLoader)
from tensorboard.compat.proto import event_pb2
from loguru import logger
import matplotlib.pyplot as plt
import torchmetrics

from ..data_utils.functions import (
    read_image, save_image, show_images_cv2, resize_image)


IntBbox = Tuple[int, int, int, int]
FloatBbox = Tuple[float, float, float, float]
Bbox = Union[IntBbox, FloatBbox]


def image_tensor_to_numpy(tensor: Tensor) -> NDArray:
    """Convert an image or a batch of images from tensor to ndarray.

    Parameters
    ----------
    tensor : Tensor
        The tensor with shape `(h, w)`, `(c, h, w)` or `(b, c, h, w)`.

    Returns
    -------
    NDArray
        The array with shape `(h, w)`, `(h, w, c)` or `(b, h, w, c)`.
    """
    if len(tensor.shape) == 3:
        return tensor.detach().cpu().permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 4:
        return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif len(tensor.shape) == 2:
        return tensor.detach().cpu().numpy()


def image_numpy_to_tensor(
    array: NDArray,
    device: torch.device = torch.device('cpu'),
    dtype: Optional[torch.dtype] = None
) -> Tensor:
    """Convert an image or a batch of images from ndarray to tensor.

    Parameters
    ----------
    array : NDArray
        The array with shape `(h, w)`, `(h, w, c)` or `(b, h, w, c)`.
    device : torch.device, optional
        Device for image tensor. By default is `torch.device('cpu')`.
    dtype : torch.dtype, optional
        Data type for image tensor. By default is `None`.

    Returns
    -------
    Tensor
        The tensor with shape `(h, w)`, `(c, h, w)` or `(b, c, h, w)`.
    """
    if len(array.shape) == 3:
        return torch.tensor(array.transpose(2, 0, 1),
                            device=device, dtype=dtype)
    elif len(array.shape) == 4:
        return torch.tensor(array.transpose(0, 3, 1, 2),
                            device=device, dtype=dtype)
    elif len(array.shape) == 2:
        return torch.tensor(array, device=device, dtype=dtype)


def draw_bounding_boxes(
    image: NDArray,
    bboxes: List[Bbox],
    class_labels: List[Union[str, int, float]] = None,
    confidences: List[float] = None,
    exclude_classes: List[Union[str, int, float]] = None,
    bbox_format: str = 'xyxy',
    line_width: Optional[int] = None,
    line_color: Tuple[int, int, int] = (255, 255, 255),
    txt_color: Tuple[int, int, int] = (255, 0, 0)
) -> NDArray:
    """Draw bounding boxes and corresponding labels on a given image.

    Parameters
    ----------
    image : NDArray
        The given image with shape `(h, w, c)`.
    bboxes : List[Bbox]
        The bounding boxes with shape `(n_boxes, 4)`.
    class_labels : List, optional
        Bounding boxes' labels. By default is None.
    confidences : List, optional
        Bounding boxes' confidences. By default is None.
    exclude_classes : List[str, int, float]
        Classes which bounding boxes won't be showed. By default is None.
    bbox_format : str, optional
        A bounding boxes' format. It should be one of "xyxy", "xywh" or
        "cxcywh". By default is 'xyxy'.
    line_width : int, optional
        A width of the bounding boxes' lines. By default is 1.
    line_color : Tuple[int, int, int], optional
        A color of the bounding boxes' lines in RGB.
        By default is `(255, 255, 255)`.
    txt_color : Tuple[int, int, int], optional
        A color of the labels' text in RGB.
        By default is `(255, 0, 0)`.

    Returns
    -------
    NDArray
        The image with drawn bounding boxes.

    Raises
    ------
    NotImplementedError
        Raise when `bbox_format` is not in `("xyxy", "xywh" and "cxcywh")`.
    """
    image = image.copy()
    if exclude_classes is None:
        exclude_classes = []

    # Convert to "xyxy"
    if bbox_format != 'xyxy':
        if bbox_format in ('xywh', 'cxcywh'):
            bboxes = box_convert(
                torch.tensor(bboxes), bbox_format, 'xyxy').tolist()
        else:
            raise NotImplementedError(
                'Implemented only for "xyxy", "xywh" and "cxcywh"'
                'bounding boxes formats.')

    line_width = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)
    font_thickness = max(line_width - 1, 1)
    font_scale = line_width / 3
    
    for i, bbox in enumerate(bboxes):
        # Check if exclude
        if class_labels is not None and class_labels[i] in exclude_classes:
            continue

        # Draw bbox
        bbox = list(map(int, bbox))  # convert float bbox to int if needed
        x1, y1, x2, y2 = bbox
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.rectangle(image, p1, p2, color=line_color, thickness=line_width,
                      lineType=cv2.LINE_AA)
        
        # Put text if needed
        if class_labels is not None:
            put_text = f'cls: {class_labels[i]} '
        else:
            put_text = ''
        if confidences is not None:
            put_text += f'conf: {confidences[i]:.2f}'
        if put_text != '':
            text_w, text_h = cv2.getTextSize(
                put_text, 0, fontScale=font_scale, thickness=font_thickness)[0]
            outside = p1[1] - text_h >= 3
            p2 = (p1[0] + text_w,
                  p1[1] - text_h - 3 if outside else p1[1] + text_h + 3)
            cv2.rectangle(image, p1, p2, line_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        put_text,
                        (p1[0], p1[1] - 2 if outside else p1[1] + text_h + 2),
                        0,
                        font_scale,
                        txt_color,
                        thickness=font_thickness,
                        lineType=cv2.LINE_AA)
    return image


def read_segmentation_mask(
    path: Path,
    one_hot: bool = False,
    n_classes: Optional[int] = None
) -> NDArray:
    """Read segmentation mask from a given path.

    Parameters
    ----------
    path : Path
        Path to the segmentation mask. It can be image or numpy array.
    one_hot : bool, optional
        Whether to return one-hot encoding. By default is `False`.
    n_classes : Optional[int], optional
        Number of classes. If not given then it will be calculated
        automatically based on a number of unique values in the mask.
        By default is `None`.

    Returns
    -------
    NDArray
        The segmentation mask.

    Raises
    ------
    ValueError
        Mask must be 2-d for class mask or 3-d for one-hot mask.
    """
    if path.suffix == '.npy':
        mask = np.load(path)
    else:
        mask = read_image(path)[..., 0]
    if mask.ndim not in {2, 3}:
        raise ValueError(
            'Mask must be 2-d for class mask or 3-d for one-hot mask,'
            f'but its shape is {mask.shape}.')
    # Class mask to one-hot
    if not one_hot and mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    # One-hot to class mask
    elif one_hot and mask.ndim == 2:
        if n_classes is None:
            n_classes = len(np.unique(mask))
        one_hot_mask = np.zeros((*mask.shape, n_classes), dtype=np.uint8)
        for cls in range(n_classes):
            one_hot_mask[..., cls] = mask == cls
        mask = one_hot_mask
    return mask


def convert_seg_mask_to_one_hot(
    seg_mask: LongTensor, n_classes: int, cls_dim: int = 0
) -> LongTensor:
    """Convert segmentation mask to one-hot encoding.

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
    LongTensor
        One-hot encoded mask.
    """
    # Prepare one-hot mask
    if cls_dim == -1:
        cls_dim = len(seg_mask.shape)
    one_hot = torch.zeros(
        (*seg_mask.shape[:cls_dim], n_classes, *seg_mask.shape[cls_dim + 1:]),
        dtype=torch.long)
    
    # Fill one-hot mask by iterating over classes
    for cls_id in range(n_classes):
        index = [slice(None)] * len(seg_mask.shape)
        index.insert(cls_dim, cls_id)
        one_hot[tuple(index)] = (seg_mask == cls_id)

    return one_hot


def convert_seg_mask_to_color(
    seg_mask: NDArray,
    cls_to_color: Dict[int, Tuple[int, int, int]]
) -> NDArray:
    """Convert segmentation mask to color mask.

    If `seg_mask` has class ids that are not in `cls_to_color` then they will
    be colored in black `(0, 0, 0)`.

    Parameters
    ----------
    seg_mask : NDArray
        Segmentation mask with shape `(h, w)` or `(b, h, w)` for batch.
    cls_to_color : Dict[int, Tuple[int, int, int]]
        A dict with classes as keys and colors as values.

    Returns
    -------
    NDArray
        Color mask with shape `(h, w, 3)` or `(b, h, w, 3)` for batch.
    """
    color_mask = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
    for cls_id, color in cls_to_color.items():
        color_mask[seg_mask == cls_id] = color
    return color_mask


class SaveImagesSegCallback:
    """Callback for saving images.
    
    Use it in train loop to save model predictions and targets.
    """

    def __init__(
        self,
        save_dir: Path,
        cls_to_color: Dict[int, Tuple[int, int, int]],
        only_first_image: bool = False,
        resize_shape: Optional[Tuple[int, int]] = None,
        save_stacked: bool = True
    ) -> None:
        """Initialize callback for saving images.

        Parameters
        ----------
        save_dir : Path
            Directory to save images.
        cls_to_color : Dict[int, Tuple[int, int, int]]
            A dict with classes as keys and colors as values.
        only_first_image : bool, optional
            Whether to save only first image in batch. By default is `False`.
        resize_shape : Optional[Tuple[int, int]], optional
            Shape to resize images before saving. By default is `None`.
        save_stacked : bool, optional
            Whether to save stacked predicts, images and labels instead of
            separate ones. By default is `True`.
        """
        self.only_first_image = only_first_image
        self.save_dir = save_dir
        self.resize_shape = resize_shape
        self.save_stacked = save_stacked
        self.cls_to_color = cls_to_color

    def __call__(
        self,
        batch: Tuple,
        predicts: Tensor,
        epoch: int,
        step: int,
    ) -> None:
        """Save images.

        Parameters
        ----------
        batch : Tuple
            Batch of images, masks, image paths, mask paths and shapes.
        predicts : Tensor
            Model predictions with shape `(b, c, h, w)`.
        epoch : int
            Current epoch number.
        step : int
            Current step number.
        """
        images, masks, img_paths, mask_paths, shapes = batch

        # Prepare inputs and predictions
        if self.only_first_image:
            images = images[0][None, ...]
            masks = masks[0][None, ...]
            predicts = predicts[0][None, ...]
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        if masks.shape[1] == 1:
            masks = masks.squeeze().cpu().numpy()
        else:
            masks = masks.argmax(dim=1).cpu().numpy()
        images = (images * 255).astype(np.uint8)
        masks = convert_seg_mask_to_color(masks, self.cls_to_color)
        predicts = convert_seg_mask_to_color(predicts, self.cls_to_color)

        # Save images
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i]
            predict = predicts[i]
            sample_name = img_paths[i].stem

            if self.resize_shape is not None:
                image = resize_image(image, self.resize_shape)
                mask = resize_image(mask, self.resize_shape)
                predict = resize_image(predict, self.resize_shape)

            if self.save_stacked:
                concatenated_image = np.concatenate(
                    [mask, image, predict],
                    axis=1)
                save_image(
                    concatenated_image,
                    self.save_dir / f'{epoch}_{step}_{sample_name}.jpg'
                )
            else:
                save_image(
                    image,
                    self.save_dir / f'{epoch}_{step}_{sample_name}_image.jpg'
                )
                save_image(
                    mask,
                    self.save_dir / f'{epoch}_{step}_{sample_name}_mask.jpg'
                )
                save_image(
                    predict,
                    self.save_dir / f'{epoch}_{step}_{sample_name}_predict.jpg'
                )
            
            if self.only_first_image:
                break


class ShowPredictCallback:
    """Callback for showing model predictions during training.
    
    Use it in train loop to show model predictions.
    """

    def __init__(
        self,
        cls_to_color: Dict[int, Tuple[int, int, int]],
        wait_time: int = 1,
        only_first_image: bool = False,
        resize_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize callback for showing model predictions.

        Parameters
        ----------
        cls_to_color : Dict[int, Tuple[int, int, int]]
            A dict with classes as keys and colors as values.
        wait_time : int, optional
            Time in milliseconds to wait before continuing. By default is `1`.
            If `0` then the image will wait until an any key is pressed.
        only_first_image : bool, optional
            Whether to show only first image in batch. By default is `False`.
        resize_shape : Optional[Tuple[int, int]], optional
            Shape to resize images before showing. By default is `None`.
        """
        self.wait_time = wait_time
        self.resize_shape = resize_shape
        self.only_first_image = only_first_image
        self.cls_to_color = cls_to_color

    def __call__(
        self,
        batch: Tuple,
        predicts: Tensor,
        epoch: int,
        step: int,
    ) -> None:
        """Save images.

        Parameters
        ----------
        batch : Tuple
            Batch of images, masks, image paths, mask paths and shapes.
        predicts : Tensor
            Model predictions with shape `(b, c, h, w)`.
        epoch : int
            Current epoch number.
        step : int
            Current step number.
        """
        images, masks, img_paths, mask_paths, shapes = batch
        
        # Prepare inputs and predictions
        if self.only_first_image:
            images = images[0][None, ...]
            masks = masks[0][None, ...]
            predicts = predicts[0][None, ...]
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        if masks.shape[1] == 1:
            masks = masks.squeeze().cpu().numpy()
        else:
            masks = masks.argmax(dim=1).cpu().numpy()
        images = (images * 255).astype(np.uint8)
        masks = convert_seg_mask_to_color(masks, self.cls_to_color)
        predicts = convert_seg_mask_to_color(predicts, self.cls_to_color)

        # Show images
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i]
            predict = predicts[i]

            if self.resize_shape is not None:
                image = resize_image(image, self.resize_shape)
                mask = resize_image(mask, self.resize_shape)
                predict = resize_image(predict, self.resize_shape)

            show_images_cv2(
                [image, mask, predict], ['image', 'mask', 'predict'],
                delay=self.wait_time, destroy_windows=False
            )


def random_crop(
    image: Union[Tensor, NDArray],
    min_size: Union[int, Tuple[int, int]],
    max_size: Union[int, Tuple[int, int]],
    return_position: bool = False
) -> Union[Tensor, NDArray, Tuple[Tensor, IntBbox], Tuple[NDArray, IntBbox]]:
    """Make random crop from a given image.

    If `min_size` is `int` then the crop will be a square with shape belonging
    to the range from `(min_size, min_size)` to `(max_size, max_size)`.
    If `min_size` is `tuple` then the crop will be a rectangle with shape
    belonging to the range from `min_size` to `max_size`.

    Parameters
    ----------
    image : Union[Tensor, NDArray]
        The given image for crop with shape `(c, h w)` for `Tensor` type and
        `(h, w, c)` for `NDArray`. `Image` can be a batch of images with shape
        `(b, c, h, w)` or `(b, h, w, c)` depending on its type.
    min_size : Union[int, Tuple[int, int]]
        Minimum size of crop. It should be either min size of square as `int`
        or min size of rectangle as `tuple` in format `(h, w)`.
    max_size : Union[int, Tuple[int, int]]
        Maximum size of crop. It should be either max size of square as `int`
        or max size of rectangle as `tuple` in format `(h, w)`.
    return_position : bool, optional
        Whether to return bounding box of made crop. By default is `False`.

    Returns
    -------
    Union[Tensor, NDArray, Tuple[Tensor, IntBbox], Tuple[NDArray, IntBbox]]
        The crop region in the same type as the original image and it's
        bounding box if `return_position` is `True`.

    Raises
    ------
    ValueError
        "image" must be 3-d for one instance and 4-d for a batch.
    ValueError
        "image" must be either "torch.Tensor" or "numpy.ndarray".
    TypeError
        "min_size" and "max_size" must be int or Tuple[int, int].
    """
    if len(image.shape) not in {3, 4}:
        raise ValueError(
            '"image" must be 3-d for one instance and 4-d for a batch,'
            f'but its shape is {image.shape}.')
    if isinstance(image, Tensor):
        randint = torch.randint
        crop_dims = (-2, -1)
    elif isinstance(image, np.ndarray):
        randint = np.random.randint
        crop_dims = (-3, -2)
    else:
        raise ValueError(
            '"image" must be either "torch.Tensor" or "numpy.ndarray"'
            f'but it is {type(image)}.')
    h = image.shape[crop_dims[0]]
    w = image.shape[crop_dims[1]]
    # Get random size of crop
    if isinstance(min_size, int) and isinstance(max_size, int):
        x_size = y_size = randint(min_size, max_size + 1, ())
    elif (isinstance(min_size, (tuple, list)) and len(min_size) == 2 and
          isinstance(max_size, (tuple, list)) and len(max_size) == 2):
        y_size = randint(min_size[0], max_size[0] + 1, ())
        x_size = randint(min_size[1], max_size[1] + 1, ())
    else:
        raise TypeError(
            '"min_size" and "max_size" must be int or Tuple[int, int] '
            f'but it is {min_size} and {max_size}.')
    # Get random window
    x_min = int(randint(0, w - x_size + 1, ()))
    y_min = int(randint(0, h - y_size + 1, ()))
    x_max = x_min + x_size
    y_max = y_min + y_size
    # Crop the window
    crop_indexes = [slice(None)] * len(image.shape)
    crop_indexes[crop_dims[0]] = slice(y_min, y_max)
    crop_indexes[crop_dims[1]] = slice(x_min, x_max)
    cropped = image[tuple(crop_indexes)]
    if return_position:
        return cropped, (x_min, y_min, x_max, y_max)
    else:
        return cropped
    

def make_compatible_state_dict(
    model: Module, state_dict: Dict[str, Tensor],
    return_discarded: bool = False
) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
    """Discard model-incompatible weights from `state_dict`.

    Weights that are not represented in the model
    or that are not size-compatible to the model's parameters will be
    discarded from the `state_dict`.

    Parameters
    ----------
    model : Module
        The model for which to combine `state_dict`.
    state_dict : Dict[str, Tensor]
        The dict of parameters to make compatible.
    return_discarded : bool, optional
        Whether to return discarded parameters. By default is `False`.

    Returns
    -------
    Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]
        Compatible state dict and dict containing discarded parameters
        if `return_discarded` is `True`.
    """
    model_state_dict = model.state_dict()
    if return_discarded:
        discarded_parameters = {}
    keys = list(state_dict.keys())
    for k in keys:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                discarded_param = state_dict.pop(k)
                if return_discarded:
                    discarded_parameters[k] = discarded_param
        else:
            discarded_param = state_dict.pop(k)
            if return_discarded:
                discarded_parameters[k] = discarded_param

    if return_discarded:
        return state_dict, discarded_parameters
    else:
        return state_dict
    

def read_tensorboard_events(path: str) -> List[float]:
    """Read tensorboard events from a given file.

    Parameters
    ----------
    path : str
        Path to the tensorboard event file.

    Returns
    -------
    List[float]
        List of read float values.
    """
    loader = RawEventFileLoader(path)
    events = []
    for i, raw_event in enumerate(loader.Load()):
        # Skip the first empty event
        if i == 0:
            continue
        e = event_pb2.Event.FromString(raw_event)
        # e.summary.value[0].tag
        events.append(e.summary.value[0].simple_value)
    return events


def plot_tensorboard_events(
    tensorboard_dir: Union[Path, str],
    metrics: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (16, 16),
    tight_layout: bool = True,
    plt_show: bool = False,
) -> List[plt.Axes]:
    """Plot metrics from tensorboard events files.

    Parameters
    ----------
    tensorboard_dir : Union[Path, str]
        Path to the directory with tensorboard events files.
    metrics : Optional[List[str]], optional
        List of metrics to plot. If `None` then all metrics in the directory
        will be plotted. By default is `None`.
    grid_size : Optional[Tuple[int, int]], optional
        Grid size for metrics displaying. By default is `None` which means
        that metrics will be displayed in one row.
    figsize : Tuple[int, int], optional
        Figure size. By default is `(16, 16)`.
    tight_layout : bool, optional
        Whether to make `plt.tight_layout()` in this function's calling.
        By default is `True`.
    plt_show : bool, optional
        Whether to make `plt.show()` in this function's calling. By default is
        `False`.

    Returns
    -------
    List[plt.Axes]
        List of axes with plotted metrics.

    Raises
    ------
    FileNotFoundError
        Tensorboard directory does not exist.
    ValueError
        `grid_size` must be a tuple of two integers.
    ValueError
        `grid_size` must be such that the total number of subplots is greater
        or equal to the number of plots.
    """
    if isinstance(tensorboard_dir, str):
        tensorboard_dir = Path(tensorboard_dir)
    if not tensorboard_dir.exists():
        raise FileNotFoundError(
            f'Tensorboard directory {str(tensorboard_dir)} does not exist.')
    if grid_size is not None and len(grid_size) != 2:
        raise ValueError('grid_size must be a tuple of two integers.')

    # Collect event files
    if metrics is None:
        metrics = sorted([
            path.name for path in tensorboard_dir.glob('*') if path.is_dir()])
    event_files: Dict[str, Dict[str, str]] = {}
    for metric in metrics:
        event_files[metric] = {}
        event_dirs = sorted(tensorboard_dir.glob(f'{metric}*'))
        for event_dir in event_dirs:
            event_file = list(event_dir.glob('*'))
            if len(event_file) != 1:
                logger.warning(
                    f'Found {len(event_file)} event files in {event_dir} '
                    'but expected 1.')
            event_files[metric][event_dir.stem] = str(event_file[0])

    # Create plot axes
    num_plots = len(metrics)
    if grid_size is None:
        # Display metrics in one row
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]
    else:
        # Display metrics in a grid
        rows, cols = grid_size
        if rows * cols < num_plots:
            raise ValueError(
                f'grid_size {grid_size} must be such that the total number '
                'of subplots is greater or equal '
                f'to the number of plots {num_plots}.')
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

    # Plot metrics
    for i, metric in enumerate(metrics):
        axes[i].set_title(metric)
        for metric_name, event_file in event_files[metric].items():
            events = read_tensorboard_events(event_file)
            axes[i].plot(events, label=metric_name)
        axes[i].legend()

    # Remove unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    if tight_layout:
        plt.tight_layout()
    if plt_show:
        plt.show()
    return axes


def get_metric_class(class_name: str) -> Type[torchmetrics.Metric]:
    """Get metric class from torchmetrics.

    Parameters
    ----------
    class_name : str
        Name of the metric class.

    Returns
    -------
    Type[torchmetrics.Metric]
        Metric class.

    Raises
    ------
    ValueError
        If metric class is not found in torchmetrics.
    """
    if hasattr(torchmetrics, class_name):
        return getattr(torchmetrics, class_name)
    raise ValueError(f"Metric class {class_name} not found in torchmetrics")
