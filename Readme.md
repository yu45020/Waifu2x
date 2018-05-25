# Waifu2x

 Re-implementation on the original [waifu2x](https://github.com/nagadomi/waifu2x) in PyTorch with additional models. 
 
 ## Details
 Models are trained on rgb based on a [waifu2x's discussion](https://github.com/nagadomi/waifu2x/issues/92). 
 
 ### Loss
The loss function is weighted MSE with weights [0.29891, 0.58661, 0.11448] on RGB channels. Please see [waifu2x's source code](https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109) and a [blog post](https://blog.csdn.net/AIchipmunk/article/details/53704139).  The weights come from the image transform from RGB to gray scale.
 
 ```python
import torch
from torch import nn
from torch.nn.modules.loss import _assert_no_grad
from torch.nn.functional import _pointwise_loss


rgb_weights = [0.29891, 0.58661, 0.11448]
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

```
 
 ### Image Processing
 High resolution PNGs are cropped into 192x192 non-overlapping patches, so some parts of images are dropped. A lesson I learn is that DON'T save processed patches. I save over 2x80 thousands small patches (both high and low resolution), and I fail to open the folder. 
 
 High resolution images are loaded all at once in memory and cut into patches. Low resolution patches are also saved in  memory. They are then dumped into PyTorch's dataloader and feed into the neural net. 
 
 ````python
def get_img_grids(self, img):
        # return nested list [ [img_file, position tuple], ...]
        img_w, img_h = img.size
        count_w, count_h = img_w // self.patch_size, img_h // self.patch_size

        # (left, top, right button) square's 4 lines as coordinate
        patch_box = [(self.patch_size * i, self.patch_size * j, self.patch_size * (i + 1), self.patch_size * (j + 1))
                     for i in range(count_w) for j in range(count_h)]
        if len(patch_box) > self.max_path_per_img:
            patch_box = random.sample(patch_box, self.max_path_per_img)

        return patch_box    
        
````
 
Low resolution images are shrunk by BICUBIC and  re-encoded into JPEG format, which is very common for low quality Anime style images. Noise factor is added via changing the quality value when re-encoding image. 
 
 ```python
class ImageAugment:
    def __init__(self, shrink_size=2, noise_level=1, down_sample_method=Image.BICUBIC):
        # noise_level (int): 0: no noise; 1: 90% quality; 2:80%

        self.noise_level = noise_level
        self.shrink_size = shrink_size
        self.down_sample_method = down_sample_method

    def shrink_img(self, hr_img):
        img_w, img_h = tuple(map(lambda x: int(x / self.shrink_size), hr_img.size))
        lr_img = hr_img.resize((img_w, img_h), self.down_sample_method)
        return lr_img

    def add_jpeg_noise(self, hr_img):
        quality = 100 - 10 * self.noise_level
        lr_img = BytesIO()
        hr_img.save(lr_img, format='JPEG', quality=quality)
        lr_img.seek(0)
        lr_img = Image.open(lr_img)
        return lr_img

    def process(self, hr_patch, grid):
        hr_patch = hr_patch.crop(grid)
        lr_patch = self.shrink_img(hr_patch)
        lr_patch = self.add_jpeg_noise(lr_patch)
        return lr_patch, hr_patch
```
 
 ## Models
 

 #### ESPCN_7
Modified from [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158).  Computations are done on the low resolution images, and the  last layer is Pixel Shuffle that scale up the input image. 

A selection unit is added in between of convolutional filters.  Details on the selection unit can be found in [A Deep Convolutional Neural Network with Selection Units for Super-Resolution](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Choi_A_Deep_Convolutional_CVPR_2017_paper.pdf). But the activation function is changed to SELU. It seems quite powerful.

![ESPCN_7 Loss](./Readme_imgs/ESPCN_7_loss.png) 



 
 In Google's colaboratory environment (GPU K-80), an iteration on 50 images (96x96 -> 192x192) is around 2.8s with around 5 GB GPU memory usage. 
 
 ```python
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
        self.conv = self.conv_block(in_channels, out_channel, selection_unit=False)
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

```
 
 
 ### Waifu2x Model 
 * [upconv_7](https://github.com/nagadomi/waifu2x/blob/3c46906cb78895dbd5a25c3705994a1b2e873199/lib/srcnn.lua#L311)
 ````python
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

````
 
 
 ## TODO: 
 * [DRRN](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) (planned)
 (Note: DRRN is not realistic for CPU only usage. A modified version might be used.)
 * and find some interesting paper
 
 