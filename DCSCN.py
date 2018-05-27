import warnings
from math import exp, log, sqrt

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

warnings.simplefilter('ignore')


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


if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    img = Image.open('2.jpg')
    img_up = img.resize((2 * img.size[0], 2 * img.size[1]))
    img_up = to_tensor(img_up).unsqueeze(0)
    img = to_tensor(img).unsqueeze(0)
    model = DCSCN(color_channel=3,
                  up_scale=2,
                  feature_layers=5,
                  first_feature_filters=64,
                  last_feature_filters=32,
                  reconstruction_filters=32,
                  up_sampler_filters=32)

    import time

    a = time.time()
    out = model.forward_checkpoint(img)
    print(time.time() - a)
    upscale = out + img_up
