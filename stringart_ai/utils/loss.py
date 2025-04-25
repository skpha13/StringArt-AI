from piqa import SSIM

class SSIMLoss(SSIM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, y):
        return 1. - super().forward(x, y)