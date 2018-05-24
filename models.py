import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

warnings.simplefilter('ignore')


class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, activation, selection_unit=False):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
             activation]
        if selection_unit:
            m.append(SelectionUnit(out_channels))
        super(Conv2dLayer, self).__init__(*m)


class SelectionUnit(nn.Sequential):
    # A Deep Convolutional Neural Network with Selection Units for Super-Resolution
    def __init__(self, n_feats):
        m = [nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0),
             nn.Sigmoid()]
        super(SelectionUnit, self).__init__(*m)


# class UpSampler2(nn.Sequential):
#     def __init__(self, upscale, n_feats, out_channels, kernel_size):
#         m = []
#         if (upscale & (upscale - 1)) == 0:  # is scale == 2^n
#             for i in range(int(math.log(upscale, base=2))):
#                 m.append(nn.Conv2d(n_feats, out_channels * 2 ** 2, kernel_size,
#                                    padding=(kernel_size - 1) // 2))
#                 m.append(nn.PixelShuffle(2))
#
#         elif upscale == 3:
#             m.append(nn.Conv2d(n_feats, out_channels * 3 ** 2, kernel_size,
#                                padding=(kernel_size - 1) // 2))
#             m.append(nn.PixelShuffle(3))
#
#         super(UpSampler2, self).__init__(*m)


class UpSampler(nn.Sequential):
    def __init__(self, upscaler, n_feats, out_channels, kernel_size, selection_unit=False):
        m = [nn.Conv2d(n_feats, out_channels * upscaler ** 2, kernel_size, padding=(kernel_size - 1) // 2),
             nn.PixelShuffle(upscaler)]
        if selection_unit:
            su = [nn.SELU(inplace=True),
                  SelectionUnit(out_channels)]
            m += su
        super(UpSampler, self).__init__(*m)


class ESPCN_7(nn.Module):
    # modified from ESPN
    def __init__(self, in_channels=3, upscale=2):
        super(ESPCN_7, self).__init__()
        out_channel = 256
        self.conv = self.conv_block(in_channels, out_channel, selection_unit=True)
        self.up_sampler = UpSampler(upscale, out_channel, in_channels, 3, selection_unit=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.up_sampler(x)
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
        m = [Conv2dLayer(in_channels, 64, 3, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(64, 64, 3, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(64, 128, 3, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(128, 128, 3, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(128, 256, 3, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(256, out_channels, 3, nn.SELU(inplace=True), selection_unit=selection_unit)
             ]
        return nn.Sequential(*m)

    @staticmethod
    def conv_block2(in_channels, out_channels, selection_unit=False):
        m = [Conv2dLayer(in_channels, 64, 5, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(64, 64, 3, nn.SELU(inplace=True), selection_unit=selection_unit),
             Conv2dLayer(64, out_channels, 3, nn.SELU(inplace=True), selection_unit=selection_unit)
             ]
        return nn.Sequential(*m)


class UpConv_7(nn.Sequential):

    def __init__(self):
        m = [nn.Conv2d(3, 16, 3, 1, 0),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(16, 32, 3, 1, 0),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(32, 64, 3, 1, 0),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(64, 128, 3, 1, 0),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(128, 128, 3, 1, 0),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(128, 256, 3, 1, 0),
             nn.LeakyReLU(0.1, True),
             nn.ConvTranspose2d(256, 3, 4, 2, 3, bias=False)
             ]
        super(UpConv_7, self).__init__(*m)
