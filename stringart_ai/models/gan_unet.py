import torch
from stringart_ai.config import Config
from stringart_ai.data_scripts.data_loader import load_data
from stringart_ai.models.trainer import plot_loss, plot_test_results, train_gan
from stringart_ai.models.u_net import UNet
from torch import nn, optim


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        # 2 in_channels, one for real and one for generated
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        # x: input image, y: real or generated output image
        return self.model(torch.cat([x, y], dim=1))


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = load_data(Config.DATASET_DIR, batch_size=16)

    generator = UNet(1, 1)
    discriminator = PatchDiscriminator(in_channels=2)

    criterion_gan = nn.BCELoss()
    criterion = nn.L1Loss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    train_loss_history, validation_loss_history = train_gan(
        generator,
        discriminator,
        train_loader,
        validation_loader,
        criterion_gan,
        criterion,
        optimizer_generator,
        optimizer_discriminator,
        epochs=25,
        lambda_loss=100,
        accumulation_steps=4,
    )

    plot_loss(train_loss_history, validation_loss_history)
    plot_test_results(generator, test_loader, device=torch.device("cuda"))
