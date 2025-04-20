import torch
from stringart_ai.config import Config
from stringart_ai.data_scripts.data_loader import load_data
from stringart_ai.models.trainer import plot_loss, plot_test_results, train
from torch import nn, optim


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()

        self.double_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv2d(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList(
            [
                DoubleConv2d(in_channels, 64),
                DoubleConv2d(64, 128),
                DoubleConv2d(128, 256),
                DoubleConv2d(256, 512),
                DoubleConv2d(512, 1024),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DoubleConv2d(1024, 512),
                DoubleConv2d(512, 256),
                DoubleConv2d(256, 128),
                DoubleConv2d(128, 64),
            ]
        )

        self.deconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ]
        )

        self.pool = nn.MaxPool2d((2, 2))
        self.last_layer_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        stk = []
        for encoder in self.encoder:
            x = encoder(x)
            stk.append(x)
            x = self.pool(x)

        x = stk.pop(-1)
        for index in range(len(self.decoder)):
            encoder_output = stk.pop(-1)
            x = self.deconv[index](x)

            x = torch.cat([encoder_output, x], dim=1)
            x = self.decoder[index](x)

        return self.last_layer_conv(x)


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = load_data(Config.DATASET_DIR, batch_size=16)

    model = UNet(1, 1)
    loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    train_loss_history, validation_loss_history = train(
        model,
        train_loader,
        validation_loader,
        loss_fn,
        optimizer,
        150,
        torch.device("cuda"),
    )

    plot_loss(train_loss_history, validation_loss_history)
    plot_test_results(model, train_loader, device=torch.device("cuda"))
    plot_test_results(model, test_loader, device=torch.device("cuda"))
