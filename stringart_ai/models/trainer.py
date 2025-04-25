import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    accumulation_steps: int = 4,
    device: torch.device = None,
    scheduler=None,
):
    """Train a PyTorch model and validate it after each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.Module
        The loss function to be used during training.
    optimizer : torch.optim.Optimizer
        The optimizer to update the model's parameters.
    epochs : int
        The number of epochs for training.
    accumulation_steps : int, optional
        Number of gradient accumulation steps before performing an optimizer step. Default is 4.
    device : torch.device, optional
        The device on which to run the model (CPU or CUDA). If None, the function will automatically choose the device.
    scheduler : torch.optim.lr_scheduler optional
        Learning rate scheduler to adjust the learning rate after each epoch.

    Returns
    -------
    tuple
        A tuple containing two lists:
            - train_loss_history (list): List of training loss values at each epoch.
            - val_loss_history (list): List of validation loss values at each epoch.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)

        # training phase
        model.train()
        running_train_loss = 0.0

        for index, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            if (index + 1) % accumulation_steps == 0 or (index + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        # validation Phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f}", end=" | ")
        print(f"Val   Loss: {epoch_val_loss:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.8f}")

        if scheduler:
            scheduler.step()

    return train_loss_history, val_loss_history


def plot_loss(train_loss, val_loss, path: str | None = None):
    """Plot the training and validation loss over epochs.

    Parameters
    ----------
    train_loss : list
        List of training loss values for each epoch.
    val_loss : list
        List of validation loss values for each epoch.
    path: str | None
        Directory path where to save output. Default is None, meaning output will not be saved.

    Returns
    -------
    None
        This function does not return any value. It directly displays the plot.
    """

    plt.title("Training & Validation Loss")
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    if path is not None:
        plt.savefig(os.path.join(path, "loss.png"))
    plt.show()


def plot_test_results(model, test_loader, device, num_images=5, path: str | None = None):
    """Plot a set of test results, displaying input images, predicted outputs, and ground truth labels.

    Parameters
    ----------
    model : torch.nn.Module
       The trained model used to generate predictions.
    test_loader : torch.utils.data.DataLoader
       DataLoader for the test dataset.
    device : torch.device
       The device on which the model is running (CPU or CUDA).
    num_images : int, optional
       The number of images to display. Default is 5.
    path: str | None
        Directory path where to save output. Default is None, meaning output will not be saved.

    Returns
    -------
    None
       This function does not return any value. It directly displays the plot.
    """

    model.eval()

    with torch.no_grad():
        # get first batch
        sample_inputs, sample_labels = next(iter(test_loader))
        sample_inputs, sample_labels = sample_inputs.to(device), sample_labels.to(device)

        # get model outputs
        sample_outputs = model(sample_inputs)

        # number of images to display
        batch_size = sample_inputs.size(0)
        num_images = min(num_images, batch_size)

        fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
        for i in range(num_images):
            # input
            axes[i, 0].imshow(sample_inputs[i].cpu().squeeze(0), cmap="gray")
            axes[i, 0].set_title(f"Input Image {i + 1}")
            axes[i, 0].axis("off")

            # predicted
            axes[i, 1].imshow(sample_outputs[i].cpu().squeeze(0), cmap="gray")
            axes[i, 1].set_title(f"Predicted Image {i + 1}")
            axes[i, 1].axis("off")

            # ground truth
            axes[i, 2].imshow(sample_labels[i].cpu().squeeze(0), cmap="gray")
            axes[i, 2].set_title(f"Ground Truth {i + 1}")
            axes[i, 2].axis("off")

        plt.tight_layout()

        if path is not None:
            plt.savefig(os.path.join(path, "predictions.png"))
        plt.show()
