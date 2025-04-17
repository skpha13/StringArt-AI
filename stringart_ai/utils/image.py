import math
import os
from typing import List, Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_float64
from stringart.utils.image import crop_image
from stringart.utils.types import CropMode


def calculate_aspect_preserved_size(image: np.ndarray, target_short_side_length: int) -> Tuple[int, int]:
    """Calculates new image dimensions preserving the aspect ratio
    based on the given target length for the shorter side.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    target_short_side_length : int
        The desired size for the shorter side of the image.

    Returns
    -------
    Tuple[int, int]
        The new height and width of the image with preserved aspect ratio.
    """

    height, width = image.shape

    if height < width:
        new_height = target_short_side_length
        aspect_ratio = new_height / height
        new_width = math.ceil(width * aspect_ratio)
    else:
        new_width = target_short_side_length
        aspect_ratio = new_width / width
        new_height = math.ceil(height * aspect_ratio)

    return new_height, new_width


def rbg2gray_inplace(images: List[np.ndarray]) -> None:
    """Converts RGB images in a list to grayscale in-place.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images. Each image can be either grayscale or RGB.
    """

    for index in range(len(images)):
        if len(images[index].shape) > 2:
            images[index] = rgb2gray(images[index])


def get_shortest_side(images: List[np.ndarray]) -> int:
    """Finds the shortest side (height or width) among a list of images.

    Parameters
    ----------
    images : List[np.ndarray]
        List of image arrays.

    Returns
    -------
    int
        The length of the shortest side across all images.
    """

    shortest_side = 2**30
    for image in images:
        shape = image.shape
        shortest_side = min(shortest_side, *shape)

    return shortest_side


def filter_images_by_minsize(images: List[np.ndarray], minsize: int = 256) -> List[np.ndarray]:
    """Filters out images that are smaller than the specified minimum size
    in either dimension.

    Parameters
    ----------
    images : List[np.ndarray]
        List of image arrays.
    minsize : int, optional
        Minimum allowed size for both dimensions (default is 256).

    Returns
    -------
    List[np.ndarray]
        Filtered list of images meeting the size requirement.
    """

    filtered = [img for img in images if img.shape[0] >= minsize and img.shape[1] >= minsize]

    return filtered


def preprocess_image_dimensions(
    images: List[np.ndarray], crop_mode: CropMode = "first-half", new_res: int = 256
) -> np.ndarray:
    """Preprocesses a list of images: converts to grayscale, filters by size,
    resizes preserving aspect ratio, and crops. Resulting images will all have the same size.

    Parameters
    ----------
    images : List[np.ndarray]
        List of image arrays to preprocess.
    crop_mode : CropMode, optional
        Cropping mode to use during cropping (default is "first-half").
    new_res : int, optional
        Target resolution for the shorter side (default is 256).

    Returns
    -------
    np.ndarray
        Array of preprocessed images.
    """

    rbg2gray_inplace(images)
    images = filter_images_by_minsize(images, new_res)

    for index in range(len(images)):
        new_height, new_width = calculate_aspect_preserved_size(images[index], new_res)
        temp_image = resize(images[index], (new_height, new_width))
        images[index] = crop_image(temp_image, crop_mode)

    return np.array(images)


def load_images(input_dir: str) -> List[np.ndarray]:
    """Loads and inverts images from a specified directory.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing image files.

    Returns
    -------
    List[np.ndarray]
        List of loaded and inverted images as float64 arrays.
    """

    image_extensions = (".png", ".jpg", ".jpeg")
    images: List[np.ndarray] = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(input_dir, filename)

            image = img_as_float64(imread(filepath))
            image = 1 - image

            if image is not None:
                images.append(image)
            else:
                print(f"Warning: Failed to load image at: {filepath}")

    return images
