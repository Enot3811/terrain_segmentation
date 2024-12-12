"""Functions for data processing without using PyTorch.

Contains utilities for:
    - Image loading, saving, processing and visualization
    - Numpy arrays loading and processing
    - File paths processing
"""


from typing import Union, List, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import cv2
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS: List[str] = ['jpg', 'jpeg', 'png']


def read_image(
    path: Union[Path, str], grayscale: bool = False, bgr_to_rgb: bool = True
) -> NDArray:
    """Read the image to a numpy array.

    Parameters
    ----------
    path : Union[Path, str]
        A path to the image file.
    grayscale : bool, optional
        Whether read image in grayscale (1-ch image), by default is `False`.
    bgr_to_rgb : bool, optional
        Whether to convert read BGR image to RGB. By default is `True`.
        Ignored if `grayscale` is `True`.

    Returns
    -------
    NDArray
        The array containing the read image.

    Raises
    ------
    ValueError
        Raise when cv2 could not read the image.
    """
    path = prepare_path(path)
    color_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), color_flag)
    if img is None:
        raise ValueError('cv2 could not read the image.')
    if grayscale:
        img = img[..., None]  # Add chanel dimension
    elif bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img


def resize_image(
    image: NDArray, new_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> NDArray:
    """Resize image to a given size.

    Parameters
    ----------
    image : NDArray
        The image to resize.
    new_size : Tuple[int, int]
        The requested size in `(h, w)` format.
    interpolation : int, optional
        cv2 interpolation flag. By default equals `cv2.INTER_LINEAR`.

    Returns
    -------
    NDArray
        The resized image

    Raises
    ------
    ValueError
        Raise when got incorrect size.
    """
    if len(new_size) != 2 or new_size[0] <= 0 or new_size[1] <= 0:
        raise ValueError(
            f'New size is required to be "(h, w)" but got {new_size}.')
    # Reverse to (w, h) for cv2
    new_size = new_size[::-1]
    resized = cv2.resize(image, new_size, None, None, None,
                         interpolation=interpolation)
    # If resize 1-channel image, channel dimension will be lost
    if len(resized.shape) != len(image.shape) and image.shape[-1] == 1:
        resized = resized[..., None]
    return resized


def show_images_plt(
    images: Union[List[NDArray], NDArray],
    subtitles: Optional[Union[List[str], str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 16),
    grid_size: Optional[Tuple[int, int]] = None,
    colormap: Optional[str] = None,
    tight_layout: bool = True,
    plt_show: bool = False
) -> List[plt.Axes]:
    """Display image(s) on a matplotlib figure(s).

    Parameters
    ----------
    images : Union[List[NDArray], NDArray]
        Image(s) to display.
    subtitles : Optional[Union[List[str], str]], optional
        Subtitles for images. By default is `None`.
    title : Optional[str], optional
        Title for the figure. By default is `None`.
    figsize : Tuple[int, int], optional
        Figsize for pyplot figure. By default is `(16, 8)`.
    grid_size : Tuple[int, int], optional
        Grid size for images displaying. By default is `None` which means
        that images will be displayed in one row.
    colormap : Optional[str], optional
        Color map for images displaying. By default is `None`.
    tight_layout : bool, optional
        Whether to make `plt.tight_layout()` in this function's calling.
        By default is `True`.
    plt_show : bool, optional
        Whether to make `plt.show()` in this function's calling.
        By default is `False`.

    Returns
    -------
    List[plt.Axes]
        Axes with showed image(s).
    """
    # Convert single image to list for consistent processing
    if isinstance(images, np.ndarray):
        images = [images]
    if subtitles is not None:
        if isinstance(subtitles, str):
            subtitles = [subtitles] * len(images)
        elif len(subtitles) != len(images):
            raise ValueError(
                'Length of subtitles must be equal to length of images.')
    if grid_size is not None:
        if len(grid_size) != 2:
            raise ValueError('grid_size must be a tuple of two integers.')
        elif grid_size[0] * grid_size[1] < len(images):
            raise ValueError(
                'grid_size must be such that the total number of subplots '
                'is greater or equal to the number of images.')
    
    num_images = len(images)
    
    if grid_size is None:
        # Display images in one row
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        if num_images == 1:
            axes = [axes]
    else:
        # Display images in a grid
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img, cmap=colormap)
            axes[i].axis('off')
            if subtitles is not None:
                axes[i].set_title(subtitles[i])
    
    # Remove unused subplots
    for i in range(num_images, len(axes)):
        fig.delaxes(axes[i])
    
    if tight_layout:
        plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if plt_show:
        plt.show()
    return axes


def show_images_cv2(
    images: Union[NDArray, List[NDArray]],
    window_title: Union[str, List[str]] = 'image',
    destroy_windows: bool = True,
    delay: int = 0,
    rgb_to_bgr: bool = True
) -> int:
    """Display one or a few images by cv2.

    Press any key to return from function. Key's code will be returned.
    If `destroy_windows` is `True` then windows will be closed.

    Parameters
    ----------
    image : NDArray
        Image array or list of image arrays.
    window_title : Union[str, List[str]], optional
        Image window's title. If List is provided it must have the same length
        as the list of images.
    destroy_windows : bool, optional
        Whether to close windows after function's end.
    delay : int, optional
        Time in ms to wait before window closing. If `0` is passed then window
        won't be closed before any key is pressed. By default is `0`.
    rgb_to_bgr : bool, optional
        Whether to convert input images from RGB to BGR before showing.
        By default is `True`. If your images are already in BGR then sign it
        as `False`.

    Returns
    -------
    int
        Pressed key code.
    """
    key_code = -1
    if isinstance(images, (List, tuple)):
        if isinstance(window_title, str):
            one_title = True
        elif (isinstance(window_title, list) and
                len(window_title) == len(images)):
            one_title = False
        else:
            raise TypeError(
                '"window_title" must be str or List[str] with the same '
                'length as the list of images.')
        for i, image in enumerate(images):
            if one_title:
                title = f'{window_title}_{i}'
            else:
                title = window_title[i]
            if rgb_to_bgr:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(title, image)
    elif isinstance(images, np.ndarray):
        if rgb_to_bgr:
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, images)
    else:
        raise TypeError('"images" must be NDArray or List of NDArrays, '
                        f'but got {type(images)}')
    key_code = cv2.waitKey(delay)
    if destroy_windows:
        cv2.destroyAllWindows()
    return key_code


def normalize_to_image(values: NDArray) -> NDArray:
    """Convert an array containing some float values to a uint8 image.

    Parameters
    ----------
    values : NDArray
        The array with float values in range [0.0, 1.0].

    Returns
    -------
    NDArray
        The uint8 image array.
    """
    min_val = values.min()
    max_val = values.max()
    return ((values - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def save_image(img: NDArray, path: Union[Path, str], rgb_to_bgr: bool = True):
    """Save a given image to a defined path.

    Parameters
    ----------
    img : NDArray
        The saving image.
    path : Union[Path, str]
        The save path.
    rgb_to_bgr : bool, optional
        Whether to convert input images from RGB to BGR before saving.
        By default is `True`. If your images are already in BGR then sign it
        as `False`.

    Raises
    ------
    RuntimeError
        Could not save image.
    """
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), img)
    if not success:
        raise RuntimeError('Could not save image.')


def get_sliding_windows(
    source_image: np.ndarray,
    h_win: int,
    w_win: int,
    h_stride: Optional[int] = None,
    w_stride: Optional[int] = None
) -> np.ndarray:
    """Cut a given image into windows with defined shapes and stride.

    Parameters
    ----------
    source_image : np.ndarray
        The original image.
    h_win : int
        Height of the windows.
    w_win : int
        Width of the windows.
    h_stride : Optional[int], optional
        The stride of the sliding windows along height.
        If not defined it will be set by `h_win` value.
    w_stride : Optional[int], optional
        The stride of the sliding windows along width.
        If not defined it will be set by `w_win` value.

    Returns
    -------
    np.ndarray
        The cut image with shape `(n_h_win, n_w_win, h_win, w_win, c)`.
    """
    h, w = source_image.shape[:2]

    if h_stride is None:
        h_stride = h_win
    if w_stride is None:
        w_stride = w_win

    x_indexer = (
        np.expand_dims(np.arange(w_win), 0) +
        np.expand_dims(np.arange(w - w_win + 1, step=w_stride), 0).T
    )
    y_indexer = (
        np.expand_dims(np.arange(h_win), 0) +
        np.expand_dims(np.arange(h - h_win + 1, step=h_stride), 0).T
    )
    windows = source_image[y_indexer][:, :, x_indexer].swapaxes(1, 2)
    return windows


def prepare_path(path: Union[Path, str]) -> Path:
    """Check an existence of the given path and convert it to `Path`.

    Parameters
    ----------
    path : Union[Path, str]
        The given file path.

    Raises
    ------
    FileNotFoundError
        Raise when file was not found.
    """
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():
        raise FileNotFoundError(f'Did not find the file "{str(path)}".')
    return path


def rotate_rectangle(
    points: List[Tuple[int, int]], angle: float, radians: bool = True
) -> List[Tuple[int, int]]:
    """Rotate points of rectangle by given angle.

    Parameters
    ----------
    points : List[Tuple[int, int]]
        Points of rectangle. There must be 2 or 4.
    angle : float
        Angle to rotate.
    radians : bool, optional
        Whether angle given in radians. Otherwise in degrees. By default equals
        `True` (radians).

    Returns
    -------
    List[Tuple[int, int]]
        List of rotated rectangle points.
    """
    if not radians:
        angle = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)

    points = [
        (
            int(c_x + np.cos(angle) * (px - c_x) - np.sin(angle) * (py - c_y)),
            int(c_y + np.sin(angle) * (px - c_x) + np.cos(angle) * (py - c_y)))
        for px, py in points]
    return points


def collect_paths(
    image_dir: Union[str, Path],
    file_extensions: Iterable[str]
) -> List[Path]:
    """Collect all paths with given extension from given directory.

    Parameters
    ----------
    image_dir : Union[str, Path]
        Directory from which image paths will be collected.
    file_extensions : Iterable[str]
        Extension of collecting files.

    Returns
    -------
    List[Path]
        Collected image paths.
    """
    paths: List[Path] = []
    for ext in file_extensions:
        paths.extend(image_dir.glob(f'*.{ext}'))
    return paths


def read_volume(path: Union[Path, str], **read_image_kwargs) -> NDArray:
    """Read a volume from a npy or image file.

    Parameters
    ----------
    path : Union[Path, str]
        A path to the volume file.
    **read_image_kwargs : Dict[str, Any]
        Optional arguments for `read_image` if image file is being read.

    Returns
    -------
    NDArray
        The read volume.

    Raises
    ------
    ValueError
        Raise when given path does not have proper extension.
    """
    path = prepare_path(path)
    if path.suffix == '.npy':
        vol = np.load(path)
    elif path.suffix[1:] in IMAGE_EXTENSIONS:
        vol = read_image(path, **read_image_kwargs)
    else:
        raise ValueError(
            f'The file extension of the path "{str(path)}" is not proper.')
    return vol
