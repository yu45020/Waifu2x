import torch
from torch import nn
from torch.nn.modules.loss import _assert_no_grad
from torch.nn.functional import _pointwise_loss

rgb_weights = [0.29891, 0.58661, 0.11448]
# RGB have different weights
# https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class WeightedMSELoss(nn.MSELoss):
    def __init__(self, weights=rgb_weights, size_average=True, reduce=True):
        super(WeightedMSELoss, self).__init__(size_average, reduce)
        self.weights = torch.FloatTensor(weights).view(3, 1, 1)

    def forward(self, input_data, target):
        _assert_no_grad(target)
        return self.mse_weighted_loss(input_data, target, size_average=self.size_average, reduce=self.reduce)

    def mse_weighted_loss(self, input_data, target, size_average=True, reduce=True):
        return _pointwise_loss(lambda a, b: (a - b) ** 2 * self.weights.expand_as(a), torch._C._nn.mse_loss,
                               input_data, target, size_average, reduce)

