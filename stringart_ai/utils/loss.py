import torchmetrics
from torch import nn


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, x, y):
        return 1.0 - self.ssim(x, y)
