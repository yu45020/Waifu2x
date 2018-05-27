import warnings
from math import sqrt, exp, log

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
        # x = torch.clamp(x, min=0, max=1)
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
        m = [nn.Conv2d(3, 16, 3, 1, 1),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(16, 32, 3, 1, 1),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(32, 64, 3, 1, 1),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(64, 128, 3, 1, 1),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(128, 128, 3, 1, 1),
             nn.LeakyReLU(0.1, True),
             nn.Conv2d(128, 256, 3, 1, 1),
             nn.LeakyReLU(0.1, True),
             nn.ConvTranspose2d(256, 3, 4, 2, 1)
             ]
        super(UpConv_7, self).__init__(*m)


class DCSCN(nn.Module):
    def __init__(self,
                 color_channel,
                 up_scale,
                 feature_layers,
                 first_feature_filters,
                 last_feature_filters,
                 reconstruction_filters,
                 up_sampler_filters,
                 dropout_rate=0.02
                 ):
        super(DCSCN, self).__init__()
        self.total_feature_channels = 0
        self.upscale = up_scale
        self.dropout_rate = dropout_rate
        self.act_fn = nn.SELU(inplace=True)
        self.feature_block = self.make_feature_extraction_block(color_channel,
                                                                feature_layers,
                                                                first_feature_filters,
                                                                last_feature_filters)

        self.reconstruction_block = self.make_reconstruction_block(reconstruction_filters)
        self.up_sampler = self.make_upsampler(reconstruction_filters * 2, up_sampler_filters, color_channel)
        self.init_params()

    def init_params(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.normal_(i.weight, mean=0, std=1 / sqrt(i.out_channels))

    def conv_block(self, in_channel, out_channel, kernel_size):
        m = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
             nn.AlphaDropout(self.dropout_rate),
             self.act_fn]
        return nn.Sequential(*m)

    def make_feature_extraction_block(self, color_channel, num_layers, first_filters, last_filters):
        # exponential decay
        # input layer
        feature_block = [self.conv_block(color_channel, first_filters, 3)]
        # rest layer
        alpha_rate = log(first_filters / last_filters) / (num_layers - 2)
        filter_nums = [round(first_filters * exp(-alpha_rate * i)) for i in range(num_layers - 1)]
        layer_filters = [[filter_nums[i], filter_nums[i + 1], 3] for i in range(num_layers - 2)]
        feature_block.extend([self.conv_block(*x) for x in layer_filters])
        self.total_feature_channels = sum(filter_nums)
        return nn.Sequential(*feature_block)

    def make_reconstruction_block(self, num_filters):
        A = self.conv_block(self.total_feature_channels, num_filters, 1)
        B1 = self.conv_block(self.total_feature_channels, num_filters // 2, 1)
        B2 = self.conv_block(num_filters // 2, num_filters, 3)
        B = nn.Sequential(*[B1, B2])
        return nn.Sequential(*[A, B])

    def make_upsampler(self, in_channel, out_channel, color_channel):
        out = out_channel * self.upscale ** 2
        m1 = nn.Sequential(nn.Conv2d(in_channel, out, kernel_size=3, padding=1),
                           nn.AlphaDropout(self.dropout_rate))
        m2 = nn.PixelShuffle(self.upscale)
        m3 = nn.Sequential(nn.Conv2d(out_channel, color_channel, kernel_size=3, padding=1, bias=False),
                           nn.AlphaDropout(self.dropout_rate))
        return nn.Sequential(*[m1, m2, m3])

    def forward(self, x):
        feature = []
        for layer in self.feature_block.children():
            x = layer(x)
            feature.append(x)
        feature = torch.cat(feature, dim=1)
        reconstruction = [layer(feature) for layer in self.reconstruction_block.children()]
        reconstruction = torch.cat(reconstruction, dim=1)
        x = self.up_sampler(reconstruction)
        return x

    def forward_checkpoint(self, x):
        feature = []
        for layer in self.feature_block.children():
            x = checkpoint(layer, x)
            feature.append(x)
        feature = torch.cat(feature, dim=1)
        reconstruction = [checkpoint(layer, feature) for layer in self.reconstruction_block.children()]
        reconstruction = torch.cat(reconstruction, dim=1)
        x = checkpoint(self.up_sampler, reconstruction)
        return x
