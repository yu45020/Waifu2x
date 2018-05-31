# Waifu2x

 Re-implementation on the original [waifu2x](https://github.com/nagadomi/waifu2x) in PyTorch with additional models. 
 
 ## Details
 Models are trained on rgb based on a [waifu2x's discussion](https://github.com/nagadomi/waifu2x/issues/92). 
 
 ### Loss
The loss function is weighted MSE with weights [0.29891, 0.58661, 0.11448] on RGB channels. Please see [waifu2x's source code](https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109) and a [blog post](https://blog.csdn.net/AIchipmunk/article/details/53704139).  The weights come from the image transform from RGB to gray scale.
 
 L1 seems to be more robust and converges faster than MSE. Need more test. 
 
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
 


 ###
 DCSCN is very interesting as it  has relatively quick forward computation, and  both the shallow model (layerr 8) and deep model (layer 12) are quick to train.
 
 SELU is a good drop in replacement for PReLu with L1 & MSE loss. Under SELU, dropout, alpha dropout, gradient clipping and batch norm have negative impact on this model. 
 
 
 ## TODO: 
 * Rewrite: split image into pieces and dump in a model, then merge the output without "grids" effect. 
 
 * [DRRN](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) (planned)
 (Note: DRRN is not realistic for CPU only usage. A modified version might be used.)
 * and find some interesting paper
 
 