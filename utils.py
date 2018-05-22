import torch
from PIL import Image
from torchvision import transforms


class ImagePatches:
    def __init__(self, seg_size, upscale):
        self.height = None
        self.width = None
        self.scale = upscale
        self.seg_size = seg_size

    def split_img_tensor(self, img_tensor):
        assert torch.is_tensor(img_tensor)
        batch, channel, self.height, self.width = img_tensor.size()
        all_patches = []
        for w in range(0, self.width, self.seg_size):
            patches = []
            for h in range(0, self.height, self.seg_size):
                patches.append(img_tensor[:, :,
                               h:min(h + self.seg_size, self.height),
                               w:min(w + self.seg_size, self.width)])
            all_patches.append(patches)
        return all_patches

    def merge_imgs(self, list_img):
        assert isinstance(list_img, list)
        out_height = self.scale * self.height
        out_widht = self.scale * self.width
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

def convert_2_tensor(file_name):
    img_tensor = Image.open(file_name)
    img_tensor = transforms.ToTensor()(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
