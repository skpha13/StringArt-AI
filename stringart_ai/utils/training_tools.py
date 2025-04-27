import os

import torch


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): Print messages when early stopping is triggered.
            delta (float): Minimum change in monitored value to qualify as improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelCheckpoint:
    def __init__(self, save_dir="checkpoints", monitor="val_loss", mode="min", verbose=True):
        """
        Args:
            save_dir (str): Path to save models.
            monitor (str): Metric to monitor ('val_loss').
            mode (str): 'min' means lower is better, 'max' means higher is better.
            verbose (bool): Whether to print messages.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")

        self.best_model_data = None
        self.best_epoch = None

    def update(
        self,
        epoch,
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        scheduler_generator,
        current_score,
    ):
        """
        Update internal best model state if current score is better.
        """

        is_best = False

        if (self.mode == "min" and current_score < self.best_score) or (
            self.mode == "max" and current_score > self.best_score
        ):
            self.best_score = current_score
            self.best_epoch = epoch
            is_best = True

        if is_best:
            self.best_model_data = {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_generator_state_dict": optimizer_generator.state_dict(),
                "optimizer_discriminator_state_dict": optimizer_discriminator.state_dict(),
                "scheduler_generator_state_dict": scheduler_generator.state_dict(),
                self.monitor: current_score,
            }

            if self.verbose:
                print(f"[ModelCheckpoint] Updated best model at epoch {epoch} with {self.monitor}: {current_score:.4f}")

    def save(self, filename="gan_checkpoint.pth"):
        """
        Save the best model data to disk.
        """

        if self.best_model_data is None:
            print("[ModelCheckpoint] No model to save yet.")
            return

        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.best_model_data, save_path)

        if self.verbose:
            print(f"[ModelCheckpoint] Saved best model to '{save_path}' (epoch {self.best_epoch}).")
