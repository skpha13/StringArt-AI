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
    for index in range(len(images)):
        if len(images[index].shape) > 2:
            images[index] = rgb2gray(images[index])


def get_shortest_side(images: List[np.ndarray]) -> int:
    shortest_side = 2**30
    for image in images:
        shape = image.shape
        shortest_side = min(shortest_side, *shape)

    return shortest_side


def filter_images_by_minsize(images: List[np.ndarray], minsize: int = 220) -> List[np.ndarray]:
    filtered = [img for img in images if img.shape[0] >= minsize and img.shape[1] >= minsize]

    return filtered


def preprocess_image_dimensions(images: List[np.ndarray], crop_mode: CropMode = "first-half") -> np.ndarray:
    rbg2gray_inplace(images)
    images = filter_images_by_minsize(images, 220)
    shortest_side = get_shortest_side(images)

    for index in range(len(images)):
        new_height, new_width = calculate_aspect_preserved_size(images[index], shortest_side)
        temp_image = resize(images[index], (new_height, new_width))
        images[index] = crop_image(temp_image, crop_mode)

    return np.array(images)


def load_images(input_dir: str) -> List[np.ndarray]:
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
