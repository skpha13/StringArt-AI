import math
import os
from typing import List, Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.util import img_as_float64


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
