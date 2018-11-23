import torch
from torch import nn

from models import BaseModule


def Conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0,
               dilation=1, groups=1, bias=True, BN=False, activation=None):
    m = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                   padding, dilation, groups, bias)]
    if BN:
        if activation:
            m += [nn.Sequential(nn.BatchNorm2d(out_channels), activation)]
        else:
            m += [nn.Sequential(nn.BatchNorm2d(out_channels))]
    if BN is False and activation is not None:
        m += [activation]
    return m


class DSConvBlock(BaseModule):
    """ depth wise separable convolution
     A Quantization-Friendly Separable Convolution for MobileNets
    shows removing batch norm + activation after depthwise conv helps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, BN=False, activation_dep=None, activation_point=None):
        super(DSConvBlock, self).__init__()

        self.depth_wise_conv = nn.Sequential(
            *Conv_block(in_channels, in_channels, kernel_size, stride, padding,
                        dilation, in_channels, bias, BN=BN, activation=activation_dep)
        )

        self.point_wise_conv = nn.Sequential(
            *Conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                        dilation=1, bias=bias, BN=BN, activation=activation_point))

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.point_wise_conv(x)
        return x


class ResidualBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, BN=True, activation=None, expand_channel_first=True):
        super(ResidualBlock, self).__init__()
        if expand_channel_first:
            middle_channel = out_channels
        else:
            middle_channel = in_channels

        self.conv = nn.Sequential(
            DSConvBlock(in_channels, middle_channel, kernel_size, 1, padding,
                        dilation, bias, BN, activation, activation),
            DSConvBlock(middle_channel, out_channels, kernel_size, 1, padding,
                        dilation, bias, BN, activation, activation),
            DSConvBlock(out_channels, out_channels, kernel_size, stride, padding,
                        dilation, bias, BN, activation, None)
        )

        if (stride > 1) or (in_channels != out_channels):
            self.residual_conv = nn.Sequential(
                *Conv_block(in_channels, out_channels, kernel_size=1, stride=stride,
                            bias=bias, BN=BN, activation=None)
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        return x + residual


class Xception(BaseModule):
    def __init__(self, color_channel=3, act_fn=nn.LeakyReLU(0.3), upscale_size=2):
        super(Xception, self).__init__()
        self.act_fn = act_fn
        self.theta = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
        self.upsize = upscale_size
        self.upscaler = lambda x: nn.functional.interpolate(x, scale_factor=upscale_size,
                                                            mode='bilinear', align_corners=False)
        self.entry_flow_1 = self.make_entry_flow_1(color_channel, 64)
        self.entry_flow_2 = self.make_entry_flow_2(64, 256)
        self.middle_flow_1 = self.make_middle_flow(256, 256, repeat_blocks=4, rate=(2, 6))
        self.middle_flow_2 = self.make_middle_flow(256, 256, repeat_blocks=4, rate=(4, 1))
        # self.out_flow = nn.Sequential(ResidualBlock(256, 64, 3, stride=1, padding=1, dilation=1,
        #                                             bias=True, BN=False, activation=self.act_fn),
        #                               nn.Conv2d(64, color_channel, kernel_size=3, padding=1, bias=True))
        self.out_flow = nn.Sequential(ResidualBlock(256, 64, 3, stride=1, padding=1, dilation=1,
                                                    bias=True, BN=False, activation=self.act_fn),
                                      nn.Conv2d(64, color_channel, kernel_size=3, padding=1, bias=True))
        # ResidualBlock(64, color_channel, 3, stride=1, padding=1, dilation=1,
        #               bias=True, BN=False, activation=self.act_fn))

    def spatial_transform(self, x):
        size = x.size()
        theta = self.theta.expand(size[0], 2, 3)
        new_size = torch.Size((size[0], size[1], size[2] * self.upsize, size[3] * self.upsize))
        grid = nn.functional.affine_grid(theta, new_size)
        return nn.functional.grid_sample(x, grid.cuda())

    def make_entry_flow_1(self, in_channel, out_channel, bias=True, BN=False):
        m = nn.Sequential(
            ResidualBlock(in_channel, 32, 3, stride=1, padding=1,
                          dilation=1, bias=bias, BN=BN, activation=self.act_fn),
            ResidualBlock(32, out_channel, 3, stride=1, padding=1,
                          dilation=1, bias=bias, BN=BN, activation=self.act_fn)
        )
        # m = nn.Sequential(
        #     *Conv_block(in_channel, 32, 3, stride=1, padding=1,
        #                 bias=bias, BN=BN, activation=self.act_fn),
        #     *Conv_block(32, out_channel, 3, stride=1, padding=1,
        #                 bias=bias, BN=BN, activation=self.act_fn)
        # )
        return m

    def make_entry_flow_2(self, in_channel, out_channel, bias=True, BN=False):
        m = nn.Sequential(
            ResidualBlock(in_channel, 256, 3, stride=1, padding=1,
                          dilation=1, bias=bias, BN=BN, activation=self.act_fn),
            ResidualBlock(256, 256, 3, stride=1, padding=1,
                          dilation=1, bias=bias, BN=BN, activation=self.act_fn),
            ResidualBlock(256, out_channel, 3, stride=1, padding=2,  # need to change  if want out-stride of 8
                          dilation=2, bias=bias, BN=BN, activation=self.act_fn)
        )
        return m

    def make_middle_flow(self, in_channel=728, out_channel=728,
                         repeat_blocks=16, rate=(2, 4), bias=True, BN=False):
        assert repeat_blocks % 2 == 0 and in_channel == out_channel

        m = []
        # for i in range(repeat_blocks):
        #     m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=rate,
        #                            dilation=rate, bias=False, BN=True, activation=self.act_fn))

        #  Effective Use of Dilated Convolutions for Segmenting Small Object Instances in Remote Sensing Imagery
        # by Ryuhei Hamaguchi & Aito Fujita & Keisuke Nemoto & Tomoyuki Imaizumi & Shuhei Hikosaka
        for i in range(repeat_blocks // 2):
            m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=rate[0],
                                   dilation=rate[0], bias=bias, BN=BN, activation=self.act_fn))
            m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=1,
                                   dilation=1, bias=bias, BN=BN, activation=self.act_fn))
        for i in range(repeat_blocks // 2):
            m.append(ResidualBlock(out_channel, out_channel, 3, stride=1, padding=rate[1],
                                   dilation=rate[1], bias=bias, BN=BN, activation=self.act_fn))
            m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=1,
                                   dilation=1, bias=bias, BN=BN, activation=self.act_fn))
        return nn.Sequential(*m)

    def forward(self, x):
        x = self.upscaler(x)
        # x = residual
        x = self.entry_flow_1(x)
        # x = self.upscaler(x)
        # x = self.spatial_transform(x)
        x = self.entry_flow_2(x)
        x = self.middle_flow_1(x)
        x = self.middle_flow_2(x)
        x = self.out_flow(x)
        return x  # + residual

#
# model = Xception(3)
# model.total_parameters()
# import time
# from torch.optim import Adam
#
# model.cuda()
# b = torch.rand(1, 3, 48, 48).cuda()
# c = torch.randn(1, 3, 96, 96).cuda()
# criteria = nn.L1Loss()
# learning_rate = 5e-3
# weight_decay = 4e-5
# optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
# model.zero_grad()
#
# outputs = model(b)
# loss = criteria(outputs, c)
# loss.backward()
# optimizer.step()
# b.requires_grad
# b = torch.randn(1,3,3,3)
# m = nn.Conv2d(3, 1, 3, 1)
# c = m(b)
# c.requires_grad
# b.requires_grad = True
# c = nn.functional.interpolate(b, scale_factor=2)
# c.requires_grad
# d = torch.ones_like(c)
# loss = criteria(c, d)
# loss.backward()
