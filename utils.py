from itertools import islice, takewhile

import torch
from torchvision.transforms.functional import to_tensor


class ImagePatches:
    def __init__(self, seg_size, upscale):
        self.height = None
        self.width = None
        self.scale = upscale
        self.seg_size = seg_size
        self.patch_count = []

    def split_img_tensor(self, img):
        img_tensor = to_tensor(img).unsqueeze(0)
        batch, channel, self.height, self.width = img_tensor.size()
        all_patches = []
        for w in range(0, self.width, self.seg_size):
            counter = 0
            for h in range(0, self.height, self.seg_size):
                all_patches.append(img_tensor[:, :,
                                   h:min(h + self.seg_size, self.height),
                                   w:min(w + self.seg_size, self.width)])
                counter += 1
            self.patch_count.append(counter)
        return all_patches

    def merge_imgs(self, list_img):
        out_height = self.scale * self.height
        out_widht = self.scale * self.width
        list_img = self.reshape(list_img)
        new_img = torch.zeros((1, 3, out_height, out_widht))
        init_w = 0
        for patches in list_img:
            init_h = 0
            for img in patches:
                batch, channel, height, width = img.size()
                new_img[:, :, init_h:height + init_h, init_w:width + init_w] = img
                init_h += height
            init_w += width
        return new_img

    def reshape(self, img_patches):
        img_patches = iter(img_patches)
        out = list(takewhile(bool, (list(islice(img_patches, 0, i)) for i in self.patch_count)))
        return out
