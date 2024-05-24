from pytorch_msssim import SSIM

class SSIM(SSIM):
    def forward(self, X, Y):
        # [-1, 1] => [0, 1]
        X = (X + 1) / 2
        Y = (Y + 1) / 2
        return 1 - super().forward(X, Y)