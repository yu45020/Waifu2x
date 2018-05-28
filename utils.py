from itertools import islice, takewhile

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor

import pytorch_ssim


def eval_ssim(img1, img2):
    ssim_loss = pytorch_ssim.ssim(img1, img2)
    return ssim_loss.item()


class ImagePatches:
    def __init__(self, seg_size, upscale):
        self.height = None
        self.width = None
        self.scale = upscale
        self.seg_size = seg_size
        self.patch_count = []

    def split_img_tensor(self, img):
        assert isinstance(img, Image.Image)
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
        new_img = torch.ones((1, 3, out_height, out_widht))

        init_w = 0
        for patches in list_img:
            init_h = 0
            for img in patches:
                batch, channel, height, width = img.size()
                new_img[:, :, init_h:height + init_h, init_w:width + init_w] = img
                init_h += height
            init_w += self.scale * self.seg_size
        return new_img

    def reshape(self, img_patches):
        img_patches = iter(img_patches)
        out = list(takewhile(bool, (list(islice(img_patches, 0, i)) for i in self.patch_count)))
        return out


class ImagePieces:
    def __init__(self, seg_size, upscale):
        self.seg_size = seg_size
        self.scale = upscale
        self.convertor = ToPILImage(mode='RGB')
        self.width = 0
        self.height = 0
        self.grids = []

    def get_img_grids(self, pil_img):
        self.width, self.height = pil_img.size
        pieces = []
        for w in range(0, self.width, self.seg_size):
            for h in range(0, self.height, self.seg_size):
                pieces.append((h, w,
                               min(h + self.seg_size, self.height),
                               min(w + self.seg_size, self.width)))
        return pieces

    def split_img_pieces(self, img):
        assert isinstance(img, Image.Image)
        self.grids = self.get_img_grids(img)
        img_pieces = [img.crop(i) for i in self.grids]
        img_tensors = [to_tensor(i).unsqueeze(0) for i in img_pieces]
        return img_tensors

    def merge_img_pieces(self, img_pieces):
        imgs = [self.convertor(i.squeeze()) for i in img_pieces]
        grids_up = [tuple(map(lambda x: x * self.scale, i)) for i in self.grids]
        out_img = Image.new(mode="RGB", size=(self.width * 2, self.height * 2))
        for img, box in zip(imgs, grids_up):
            out_img.paste(img, box)
        out_img = to_tensor(out_img).unsqueeze(0)
        return out_img


class Check:
    def __init__(self):
        pass

    def forward_checkpoint(self, x):
        img = ToPILImage('RGB')(x.squeeze())
        img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.BICUBIC)
        img_t = to_tensor(img).unsqueeze(0)
        return img_t
