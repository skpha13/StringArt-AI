import os
from typing import List

import numpy as np
from skimage.io import imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte
from stringart.solver import Solver
from stringart.utils.image import crop_image
from stringart.utils.types import CropMode, Rasterization
from stringart_ai.utils.image import calculate_aspect_preserved_size, get_shortest_side, load_images, rbg2gray_inplace

INPUT_DIR = "../../data/sketch-subset"
OUTPUT_DIR_COMPUTED = "../../data/sketch-subset-computed"
OUTPUT_DIR_PREPROCESSED = "../../data/sketch-subset-scaled"

CROP_MODE: CropMode = "first-half"
NUMBER_OF_PEGS: int = 100
RASTERIZATION: Rasterization = "xiaolin-wu"


def compute_stringart(images: np.ndarray, path: str) -> np.ndarray:
    outputs: List[np.ndarray] = []

    for idx, image in enumerate(images):
        solver: Solver = Solver(image, crop_mode=CROP_MODE, number_of_pegs=NUMBER_OF_PEGS, rasterization=RASTERIZATION)
        A, x = solver.least_squares()
        output = solver.compute_solution(A, x)
        outputs.append(output)

        imsave(os.path.join(path, "least-squares", f"image_{idx}.png"), img_as_ubyte(output))

    return np.array(outputs)


def save_images(images: np.ndarray, path: str) -> None:
    for idx, image in enumerate(images):
        filename = os.path.join(path, f"image_{idx}.png")
        imsave(filename, img_as_ubyte(1 - image))


def preprocess_image_dimensions(images: List[np.ndarray]) -> np.ndarray:
    rbg2gray_inplace(images)
    shortest_side = get_shortest_side(images)

    for index in range(len(images)):
        new_height, new_width = calculate_aspect_preserved_size(images[index], shortest_side)
        temp_image = resize(images[index], (new_height, new_width))
        images[index] = crop_image(temp_image, CROP_MODE)

    return np.array(images)


def main():
    images = load_images(INPUT_DIR)
    images = preprocess_image_dimensions(images)
    save_images(images, OUTPUT_DIR_PREPROCESSED)

    images_stringart = compute_stringart(images, OUTPUT_DIR_COMPUTED)


if __name__ == "__main__":
    main()
