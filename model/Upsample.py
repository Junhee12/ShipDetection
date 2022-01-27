import torch
import torch.nn as nn



# ---------------------------------------------------#
#   cov + upsample
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')  # default mode='nearest'
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x
