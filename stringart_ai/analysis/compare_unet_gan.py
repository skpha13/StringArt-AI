import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from stringart_ai.config import Config
from stringart_ai.data_scripts.data_loader import load_data
from stringart_ai.models.u_net import UNet


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_unet = torch.load("../../checkpoints/best_unet_checkpoint.pth", map_location=device, weights_only=False)
    checkpoint_gan = torch.load("../../checkpoints/best_gan_checkpoint.pth", map_location=device, weights_only=False)

    unet = UNet(1, 1)
    unet.load_state_dict(checkpoint_unet["model_state_dict"])

    gan = UNet(1, 1)
    gan.load_state_dict(checkpoint_gan["generator_state_dict"])

    return unet, gan


def plot_comparative_analysis(test_images_batch, test_labels_batch, unet_outputs, gan_outputs):
    """
    Args:
        test_images_batch (Tensor): shape (N, 1, H, W)
        test_labels_batch (Tensor): shape (N, 1, H, W)
        unet_outputs (Tensor): shape (N, 1, H, W)
        gan_outputs (Tensor): shape (N, 1, H, W)
    """

    test_images_batch = test_images_batch.cpu()
    test_labels_batch = test_labels_batch.cpu()
    unet_outputs = unet_outputs.cpu()
    gan_outputs = gan_outputs.cpu()

    n_images = test_images_batch.size(0)

    fig, axes = plt.subplots(n_images, 7, figsize=(22, 3 * n_images))

    if n_images == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(n_images):
        input_img = test_images_batch[idx].squeeze().numpy()
        label_img = test_labels_batch[idx].squeeze().numpy()
        unet_pred = unet_outputs[idx].squeeze().numpy()
        gan_pred = gan_outputs[idx].squeeze().numpy()

        # Calculate SSIM and MSE
        ssim_unet = ssim(input_img, unet_pred, data_range=unet_pred.max() - unet_pred.min())
        ssim_gan = ssim(input_img, gan_pred, data_range=gan_pred.max() - gan_pred.min())

        mse_unet = mean_squared_error(input_img.flatten(), unet_pred.flatten())
        mse_gan = mean_squared_error(input_img.flatten(), gan_pred.flatten())

        # Plot Input Image
        axes[idx, 0].imshow(input_img, cmap="gray")
        axes[idx, 0].set_title("Input Image")
        axes[idx, 0].axis("off")

        # Plot Ground Truth Label
        axes[idx, 1].imshow(label_img, cmap="gray")
        axes[idx, 1].set_title("Ground Truth Label")
        axes[idx, 1].axis("off")

        # Plot UNet Prediction
        axes[idx, 2].imshow(unet_pred, cmap="gray")
        axes[idx, 2].set_title(f"UNet Prediction\nSSIM: {ssim_unet:.3f}\nMSE: {mse_unet:.3f}")
        axes[idx, 2].axis("off")

        # Plot GAN Prediction
        axes[idx, 3].imshow(gan_pred, cmap="gray")
        axes[idx, 3].set_title(f"GAN Prediction\nSSIM: {ssim_gan:.3f}\nMSE: {mse_gan:.3f}")
        axes[idx, 3].axis("off")

        # Plot Difference (Label - UNet)
        diff_unet = np.abs(label_img - unet_pred)
        axes[idx, 4].imshow(diff_unet, cmap="hot")
        axes[idx, 4].set_title("Diff: GT - UNet")
        axes[idx, 4].axis("off")

        # Plot Difference (Label - GAN)
        diff_gan = np.abs(label_img - gan_pred)
        axes[idx, 5].imshow(diff_gan, cmap="hot")
        axes[idx, 5].set_title("Diff: GT - GAN")
        axes[idx, 5].axis("off")

        # Plot Difference (UNet - GAN)
        diff_models = np.abs(unet_pred - gan_pred)
        axes[idx, 6].imshow(diff_models, cmap="hot")
        axes[idx, 6].set_title("Diff: UNet - GAN")
        axes[idx, 6].axis("off")

    plt.tight_layout()
    plt.savefig("../../plots/gan_vs_unet_plot.png")
    plt.show()


def main():
    unet, gan = load_models()
    unet.eval()
    gan.eval()

    _, _, test_loader = load_data(Config.DATASET_DIR)

    test_images = []
    test_labels = []

    for inputs, labels in test_loader:
        for image, label in zip(inputs, labels):
            test_images.append(image)
            test_labels.append(label)

            if len(test_images) == 5:
                break

        if len(test_images) == 5:
            break

    test_images_batch = torch.stack(test_images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_images_batch = test_images_batch.to(device)
    test_labels_batch = torch.stack(test_labels)

    unet = unet.to(device)
    gan = gan.to(device)

    with torch.no_grad():
        unet_outputs = unet(test_images_batch)
        gan_outputs = gan(test_images_batch)

    plot_comparative_analysis(test_images_batch, test_labels_batch, unet_outputs, gan_outputs)


if __name__ == "__main__":
    main()
