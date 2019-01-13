import torch
from torch import nn
from torch.nn.functional import _pointwise_loss

rgb_weights = [0.29891 * 3, 0.58661 * 3, 0.11448 * 3]
# RGB have different weights
# https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class WeightedHuberLoss(nn.SmoothL1Loss):
    def __init__(self, weights=rgb_weights):
        super(WeightedHuberLoss, self).__init__(size_average=True, reduce=True)
        self.weights = torch.FloatTensor(weights).view(3, 1, 1)

    def forward(self, input_data, target):
        diff = torch.abs(input_data - target)
        z = torch.where(diff < 1, 0.5 * torch.pow(diff, 2), (diff - 0.5))
        out = z * self.weights.expand_as(diff)
        return out.mean()


def weighted_mse_loss(input, target, weights):
    out = (input - target) ** 2
    out = out * weights.expand_as(out)
    loss = out.sum(0)  # or sum over whatever dimensions
    return loss / out.size(0)


class WeightedL1Loss(nn.SmoothL1Loss):
    def __init__(self, weights=rgb_weights):
        super(WeightedHuberLoss, self).__init__(size_average=True, reduce=True)
        self.weights = torch.FloatTensor(weights).view(3, 1, 1)

    def forward(self, input_data, target):
        return self.l1_loss(input_data, target, size_average=self.size_average,
                            reduce=self.reduce)

    def l1_loss(self, input_data, target, size_average=True, reduce=True):
        return _pointwise_loss(lambda a, b: torch.abs(a - b) * self.weights.expand_as(a),
                               torch._C._nn.l1_loss, input_data, target, size_average, reduce)
