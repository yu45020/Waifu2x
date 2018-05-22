import torch
import torch.nn as nn
import math
import warnings
from torch.utils.checkpoint import checkpoint
warnings.simplefilter('ignore')


class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, activation, SU=False):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
             activation]
        if SU:  # add selection unit
            # A Deep Convolutional Neural Network with Selection Units for Super-Resolution
            su = [nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
                  nn.Sigmoid()]
            m += su
        super(Conv2dLayer, self).__init__(*m)


class UpSampler2(nn.Sequential):
    def __init__(self, upscaler, n_feats, out_channels, kernel_size):
        m = []
        if (upscaler & (upscaler - 1)) == 0:  # is scale == 2^n
            for i in range(int(math.log(upscaler, base=2))):
                m.append(nn.Conv2d(n_feats, out_channels * 2 ** 2, kernel_size,
                                   padding=(kernel_size - 1) // 2))
                m.append(nn.PixelShuffle(2))

        elif upscaler == 3:
            m.append(nn.Conv2d(n_feats, out_channels * 3 ** 2, kernel_size,
                               padding=(kernel_size - 1) // 2))
            m.append(nn.PixelShuffle(3))

        super(UpSampler2, self).__init__(*m)


class UpSampler(nn.Sequential):
    def __init__(self, upscaler, n_feats, out_channels, kernel_size):
        m = [nn.Conv2d(n_feats, out_channels * upscaler ** 2, kernel_size, padding=(kernel_size - 1) // 2),
             nn.PixelShuffle(upscaler)]
        # nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)]

        super(UpSampler, self).__init__(*m)


class ESPCN_5(nn.Module):

    # modified from ESPN
    def __init__(self, in_channels=3, upscale=2):
        super(ESPCN_5, self).__init__()
        out_channel = 512
        self.conv = self.conv_block2(in_channels, out_channel, selection_unit=True)
        self.up_sampler = UpSampler(upscale, out_channel, in_channels, 3)

    def forward(self, x):
        # residual = x
        x = self.conv(x)
        x = self.up_sampler(x)
        # x = torch.add(residual, x)
        x = torch.clamp(x, min=0, max=1)
        return x

    def forward_checkpoint(self, x):
        # intermediate outputs will be dropped to save memory
        x = checkpoint(self.conv, x)
        x = checkpoint(self.up_sampler, x)
        x = torch.clamp(x, min=0, max=1)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels, selection_unit=False):
        m = [Conv2dLayer(in_channels, 32, 3, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(32, 64, 3, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(64, 128, 3, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(128, 256, 3, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(256, 256, 3, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(256, out_channels, 3, nn.SELU(inplace=True), SU=selection_unit)
             ]
        return nn.Sequential(*m)

    @staticmethod
    def conv_block2(in_channels, out_channels, selection_unit=False):
        m = [Conv2dLayer(in_channels, 64, 5, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(64, 64, 3, nn.SELU(inplace=True), SU=selection_unit),
             Conv2dLayer(64, out_channels, 3, nn.SELU(inplace=True), SU=selection_unit)
             ]
        return nn.Sequential(*m)
