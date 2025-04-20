# Data Pipeline

## Input Data

The dataset used for training the string art AI model is based on the [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) dataset, which contains a total of `50,000` sketch-style images. These images are organized into `1,000` subfolders, with each subfolder representing a unique class, and each containing `50` images.

## Data Preparation

The data preparation process involves the following steps:

1. **Image Extraction**: I begin by extracting all images from each class-specific subfolder within the dataset.

2. **Resolution Filtering**: Each image is checked for its resolution. Any image with dimensions smaller than `256x256` pixels is discarded to ensure consistent image quality across the dataset.

3. **Image Resizing**: The remaining images are resized to a uniform resolution of `256x256` pixels to standardize the input for model training.

## Label Computation

After the images have been prepared to be computed, I create my `Solver` object. This object contains all necessary methods for computing the string art configuration for each image.

### Configuration

The following configuration was used for the label generation:

```python
CROP_MODE = "first-half"
NUMBER_OF_PEGS: int = 100
RASTERIZATION: Rasterization = "xiaolin-wu"
solver: Solver = "least-squares"
matrix_representation: MatrixRepresentation = "sparse"
```

We use the least squares method along with a sparse matrix representation, which provides an efficient solution, on average, processing each image takes around 5 seconds.

This is the fastest methods out of all the available ones.

### Processing Function

The core logic for label computation is encapsulated in the `process_single_image` function:

```python
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
```
> stringart_ai/data_scripts/stringart_labeler.py

### Parallelization

To speed up the computation process, the labeler can be parallelized using Python’s `ProcessPoolExecutor`:

```python
 with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_image, images))
```
> stringart_ai/data_scripts/stringart_labeler.py

However, **Kaggle environments tend to hang** when using multiprocessing. In such cases, a simple loop is recommended:

```python
results = [process_single_image(args) for args in indexed_images]
```
> stringart_ai/data_scripts/stringart_labeler.py

On personal machines or high-performance computing (HPC) environments, **multiprocessing is preferred** for significantly faster processing times.

### Output Structure

- **Processed images** are saved to: `DATASET_DIR/images/`
- **Computed labels** are saved to: `DATASET_DIR/labels/`
- A `metadata.csv` file mapping each image to its corresponding label is also generated inside the `DATASET_DIR`.

The directories are configured via the `.env` file:

```python
IMAGENET_SKETCH_DIR='../../data/sketch' # input data directory
DATASET_DIR='../../data/stringart-dataset' # output directory for processed images and labels
```
> .env

## Data Loader

To streamline model training, we define a custom dataset class and a helper function to prepare PyTorch `DataLoader` objects.

### `StringArtDataset`

This dataset class takes a list of `(image_path, label_path)` tuples and loads the data into memory:

```python
class StringArtDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

        self.images = []
        self.labels = []

        for img_path, label_path in self.file_list:
            img = np.array(imread(img_path))
            label = np.array(imread(label_path))

            self.images.append(to_tensor(img))
            self.labels.append(to_tensor(label))

        self.images = torch.stack(self.images)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
```
> stringart_ai/data_scripts/data_loader.py

This class loads all the images and labels into memory as PyTorch tensors, making them easily accessible during training and evaluation.

### `load_data` function

This function handles the loading and splitting of the dataset into training, validation, and test sets, and returns their respective `DataLoader`'s:

```python
def load_data(input_dir: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the dataset from the specified directory, split it into training, validation, and test sets,
    and return corresponding DataLoaders for each set.

    Parameters
    ----------
    input_dir : str
        The directory path containing the dataset. It should have a `metadata.csv` file and
        subdirectories `images` and `labels` containing the image and label files, respectively.
    batch_size : int, optional
        The batch size for the DataLoader. Default is 64.

    Returns
    -------
    tuple
        A tuple containing three `DataLoader` objects:
        - train_loader (DataLoader): The DataLoader for the training dataset.
        - validation_loader (DataLoader): The DataLoader for the validation dataset.
        - test_loader (DataLoader): The DataLoader for the test dataset.
    """

    df = pd.read_csv(os.path.join(input_dir, "metadata.csv"))
    samples = [
        (os.path.join(input_dir, "images", img_name), os.path.join(input_dir, "labels", label_name))
        for _, (img_name, label_name) in df.iterrows()
    ]

    train_data, temp_data = train_test_split(samples, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(StringArtDataset(train_data), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(StringArtDataset(validation_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(StringArtDataset(test_data), batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader
```
> stringart_ai/data_scripts/data_loader.py

### Summary

- The dataset is split in a 70/15/15 ratio for training, validation, and testing.

- All splits are loaded using PyTorch `DataLoader`'s for easy integration into model training loops.

- The input directory must contain:
    - `metadata.csv`: Mapping of image names to label names
    - `images/`: Directory containing the input images
    - `labels/`: Directory containing the corresponding output labels

## Kaggle Computing

### Motivation

Processing all **50,000** images locally proved too resource-intensive, so I turned to [Kaggle](https://www.kaggle.com/) for its free compute capabilities and convenient dataset management system. It’s a solid option for large-scale experiments, though it does come with limitations, even when not using any accelerators.

**One key constraint**: Kaggle notebooks have a **12-hour runtime limit**. Fortunately, they can continue running in the background, even after closing the session.

### `StringArt-AI 1000` Dataset

To enable faster experimentation and model prototyping, I created a smaller subset of the full dataset:

- **1,028** image-label pairs

- [Available on Kaggle](https://www.kaggle.com/datasets/adrianmincu/stringart-ai-1000)

This subset is perfect for quick testing and validation before committing to full-scale model training.

### `StringArt-AI` Full Dataset (Work in Progress)

Processing the complete dataset (**~150 hours** estimated) exceeds Kaggle’s **12-hour** compute window, so I devised a workaround by breaking it down into 10 chunks out of the 1000 folders:

- Each chunk spans 100 folder (e.g., 0–100, 101–200, ..., 901–1000)

- Each is processed independently and will later be merged into a full dataset

**Challenges**

- Kaggle only allows **5 concurrent sessions** (including background runs), so chunks `0–4` must complete before running `5–9`

- This adds manual coordination but keeps processing manageable within Kaggle’s limits
