import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

INPUT_DIR = "../../data/stringart-dataset"


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


def main():
    df = pd.read_csv(os.path.join(INPUT_DIR, "metadata.csv"))
    samples = [
        (os.path.join(INPUT_DIR, "images", img_name), os.path.join(INPUT_DIR, "labels", label_name))
        for _, (img_name, label_name) in df.iterrows()
    ]

    train_data, temp_data = train_test_split(samples, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(StringArtDataset(train_data), batch_size=64, shuffle=True)
    validation_loader = DataLoader(StringArtDataset(validation_data), batch_size=64, shuffle=False)
    test_loader = DataLoader(StringArtDataset(test_data), batch_size=64, shuffle=False)

    batch = next(iter(train_loader))
    img, label = batch[0][0], batch[1][0]

    image = img.cpu().numpy().transpose(1, 2, 0)
    label = label.cpu().numpy().transpose(1, 2, 0)

    fig, axs = plt.subplots(1, 2, figsize=[2 * 6.4, 4.8], constrained_layout=True)
    cmap = "gray"

    axs[0].imshow(image, cmap=cmap)
    axs[0].set_axis_off()

    axs[1].imshow(label, cmap=cmap)
    axs[1].set_axis_off()

    fig.show()


if __name__ == "__main__":
    main()
