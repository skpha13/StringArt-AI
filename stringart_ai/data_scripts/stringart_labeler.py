import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.util import img_as_ubyte
from stringart.solver import Solver
from stringart.utils.types import CropMode, Rasterization
from stringart_ai.utils.image import load_images, preprocess_image_dimensions
from tqdm import tqdm

INPUT_DIR = "../../data/sketch"
OUTPUT_DIR = "../../data/stringart-dataset"

CROP_MODE: CropMode = "first-half"
NUMBER_OF_PEGS: int = 100
RASTERIZATION: Rasterization = "xiaolin-wu"

IDX: int = 0

df = pd.DataFrame({"image": [], "label": []})


def process_single_image(args: Tuple[int, np.ndarray, str]) -> Tuple[str, str]:
    idx, image, path = args
    solver: Solver = Solver(image, crop_mode=CROP_MODE, number_of_pegs=NUMBER_OF_PEGS, rasterization=RASTERIZATION)
    A, x = solver.least_squares()
    output = solver.compute_solution(A, x)

    image_name = f"image_{idx:06}.png"
    label_name = f"label_{idx:06}.png"
    imsave(os.path.join(path, "images", image_name), img_as_ubyte(1 - image))
    imsave(os.path.join(path, "labels", label_name), img_as_ubyte(output))

    return image_name, label_name


def output_stringart(images: np.ndarray, path: str) -> None:
    global IDX, df

    indexed_images = [(IDX + i, image, path) for i, image in enumerate(images)]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_image, indexed_images))

    # kaggle version
    # results = [process_single_image(args) for args in indexed_images]

    IDX += len(images)

    df = pd.concat([df, pd.DataFrame(results, columns=["image", "label"])], ignore_index=True)


def process_images():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

    files = sorted(os.listdir(INPUT_DIR))

    for file in tqdm(files, desc="Processing Folders"):
        filepath = os.path.join(INPUT_DIR, file)

        if not os.path.isdir(filepath):
            continue

        image_batch = load_images(filepath)
        image_batch_preprocessed = preprocess_image_dimensions(image_batch, crop_mode=CROP_MODE, new_res=256)

        output_stringart(image_batch_preprocessed, OUTPUT_DIR)


def main():
    global df

    start_time = time.perf_counter()
    process_images()
    end_time = time.perf_counter()

    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = execution_time % 60

    print(f"Execution time:\n\t{hours} hours\n\t{minutes} minutes\n\t{seconds:.4f} seconds")


if __name__ == "__main__":
    main()
