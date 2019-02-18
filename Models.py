import json
from collections import OrderedDict
from math import exp

from Common import *


# warnings.simplefilter('ignore')

# +++++++++++++++++++++++++++++++++++++
#           DCSCN
# -------------------------------------

class DCSCN(BaseModule):
    # https://github.com/jiny2001/dcscn-super-resolution
    def __init__(self,
                 color_channel=3,
                 up_scale=2,
                 feature_layers=12,
                 first_feature_filters=196,
                 last_feature_filters=48,
                 reconstruction_filters=128,
                 up_sampler_filters=32
                 ):
        super(DCSCN, self).__init__()
        self.total_feature_channels = 0
        self.total_reconstruct_filters = 0
        self.upscale = up_scale

        self.act_fn = nn.SELU(inplace=False)
        self.feature_block = self.make_feature_extraction_block(color_channel,
                                                                feature_layers,
                                                                first_feature_filters,
                                                                last_feature_filters)

        self.reconstruction_block = self.make_reconstruction_block(reconstruction_filters)
        self.up_sampler = self.make_upsampler(up_sampler_filters, color_channel)
        self.selu_init_params()

    def selu_init_params(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                i.weight.data.normal_(0.0, 1.0 / sqrt(i.weight.numel()))
                if i.bias is not None:
                    i.bias.data.fill_(0)

    def conv_block(self, in_channel, out_channel, kernel_size):
        m = OrderedDict([
            # ("Padding", nn.ReplicationPad2d((kernel_size - 1) // 2)),
            ('Conv2d', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)),
            ('Activation', self.act_fn)
        ])

        return nn.Sequential(m)

    def make_feature_extraction_block(self, color_channel, num_layers, first_filters, last_filters):
        # input layer
        feature_block = [("Feature 1", self.conv_block(color_channel, first_filters, 3))]
        # exponential decay
        # rest layers
        alpha_rate = log(first_filters / last_filters) / (num_layers - 1)
        filter_nums = [round(first_filters * exp(-alpha_rate * i)) for i in range(num_layers)]

        self.total_feature_channels = sum(filter_nums)

        layer_filters = [[filter_nums[i], filter_nums[i + 1], 3] for i in range(num_layers - 1)]

        feature_block.extend([("Feature {}".format(index + 2), self.conv_block(*x))
                              for index, x in enumerate(layer_filters)])
        return nn.Sequential(OrderedDict(feature_block))

    def make_reconstruction_block(self, num_filters):
        B1 = self.conv_block(self.total_feature_channels, num_filters // 2, 1)
        B2 = self.conv_block(num_filters // 2, num_filters, 3)
        m = OrderedDict([
            ("A", self.conv_block(self.total_feature_channels, num_filters, 1)),
            ("B", nn.Sequential(*[B1, B2]))
        ])
        self.total_reconstruct_filters = num_filters * 2
        return nn.Sequential(m)

    def make_upsampler(self, out_channel, color_channel):
        out = out_channel * self.upscale ** 2
        m = OrderedDict([
            ('Conv2d_block', self.conv_block(self.total_reconstruct_filters, out, kernel_size=3)),
            ('PixelShuffle', nn.PixelShuffle(self.upscale)),
            ("Conv2d", nn.Conv2d(out_channel, color_channel, kernel_size=3, padding=1, bias=False))
        ])

        return nn.Sequential(m)

    def forward(self, x):
        # residual learning
        lr, lr_up = x
        feature = []
        for layer in self.feature_block.children():
            lr = layer(lr)
            feature.append(lr)
        feature = torch.cat(feature, dim=1)

        reconstruction = [layer(feature) for layer in self.reconstruction_block.children()]
        reconstruction = torch.cat(reconstruction, dim=1)

        lr = self.up_sampler(reconstruction)
        return lr + lr_up


# +++++++++++++++++++++++++++++++++++++
#           CARN      
# -------------------------------------

class CARN_Block(BaseModule):
    def __init__(self, channels, kernel_size=3, padding=1, dilation=1,
                 groups=1, activation=nn.SELU(), repeat=3,
                 SEBlock=False, conv=nn.Conv2d,
                 single_conv_size=1, single_conv_group=1):
        super(CARN_Block, self).__init__()
        m = []
        for i in range(repeat):
            m.append(ResidualFixBlock(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                      groups=groups, activation=activation, conv=conv))
            if SEBlock:
                m.append(SpatialChannelSqueezeExcitation(channels, reduction=channels))
        self.blocks = nn.Sequential(*m)
        self.singles = nn.Sequential(
            *[ConvBlock(channels * (i + 2), channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(repeat)])

    def forward(self, x):
        c0 = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)

        return x


class CARN(BaseModule):
    # Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
    # https://github.com/nmhkahn/CARN-pytorch
    def __init__(self,
                 color_channels=3,
                 mid_channels=64,
                 scale=2,
                 activation=nn.SELU(),
                 num_blocks=3,
                 conv=nn.Conv2d):
        super(CARN, self).__init__()

        self.color_channels = color_channels
        self.mid_channels = mid_channels
        self.scale = scale

        self.entry_block = ConvBlock(color_channels, mid_channels, kernel_size=3, padding=1, activation=activation,
                                     conv=conv)
        self.blocks = nn.Sequential(
            *[CARN_Block(mid_channels, kernel_size=3, padding=1, activation=activation, conv=conv,
                         single_conv_size=1, single_conv_group=1)
              for _ in range(num_blocks)])
        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=1, padding=0,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])

        self.upsampler = UpSampleBlock(mid_channels, scale=scale, activation=activation, conv=conv)
        self.exit_conv = conv(mid_channels, color_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry_block(x)
        c0 = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out


class CARN_V2(CARN):
    def __init__(self, color_channels=3, mid_channels=64, scale=2, activation=nn.LeakyReLU(0.1),
                 SEBlock=False, conv=nn.Conv2d, atrous=(1, 1, 1), repeat_blocks=3,
                 single_conv_size=1, single_conv_group=1):
        super(CARN_V2, self).__init__(color_channels=color_channels, mid_channels=mid_channels, scale=scale,
                                      activation=activation, conv=conv)

        num_blocks = len(atrous)
        m = []
        for i in range(num_blocks):
            m.append(CARN_Block(mid_channels, kernel_size=3, padding=1, dilation=1,
                                activation=activation, SEBlock=SEBlock, conv=conv, repeat=repeat_blocks,
                                single_conv_size=single_conv_size, single_conv_group=single_conv_group))
            # m.append(ResidualFixBlock(mid_channels, mid_channels, kernel_size=3, padding=atrous[i], dilation=atrous[i],
            #                           groups=1, activation=activation, conv=conv))
        self.blocks = nn.Sequential(*m)

        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])

    def forward(self, x):
        x = self.entry_block(x)
        c0 = x
        res = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = x + res
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out

    # def forward(self, x):
    #     res = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    #     out = super().forward(x)
    #     return res + out


# +++++++++++++++++++++++++++++++++++++
#           original Waifu2x model
# -------------------------------------


class UpConv_7(BaseModule):
    # https://github.com/nagadomi/waifu2x/blob/3c46906cb78895dbd5a25c3705994a1b2e873199/lib/srcnn.lua#L311
    def __init__(self):
        super(UpConv_7, self).__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7  # because of 0 padding
        from torch.nn import ZeroPad2d
        self.pad = ZeroPad2d(self.offset)
        m = [nn.Conv2d(3, 16, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(16, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 256, 3, 1, 0),
             self.act_fn,
             # in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=
             nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False)
             ]
        self.Sequential = nn.Sequential(*m)

    def load_pre_train_weights(self, json_file):
        with open(json_file) as f:
            weights = json.load(f)
        box = []
        for i in weights:
            box.append(i['weight'])
            box.append(i['bias'])
        own_state = self.state_dict()
        for index, (name, param) in enumerate(own_state.items()):
            own_state[name].copy_(torch.FloatTensor(box[index]))

    def forward(self, x):
        x = self.pad(x)
        return self.Sequential.forward(x)

    def forward_checkpoint(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out


class Vgg_7(UpConv_7):
    def __init__(self):
        super(Vgg_7, self).__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7
        m = [nn.Conv2d(3, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 3, 3, 1, 0)
             ]
        self.Sequential = nn.Sequential(*m)
