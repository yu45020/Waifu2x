import copy
import glob
import os
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image
from torchvision.transforms.functional import to_tensor

from Models import *


class ImageSplitter:
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, seg_size=48, scale_factor=2, boarder_pad_size=3):
        self.seg_size = seg_size
        self.scale_factor = scale_factor
        self.pad_size = boarder_pad_size
        self.height = 0
        self.width = 0
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def split_img_tensor(self, pil_img, scale_method=Image.BILINEAR, img_pad=0):
        # resize image and convert them into tensor
        img_tensor = to_tensor(pil_img).unsqueeze(0)
        img_tensor = nn.ReplicationPad2d(self.pad_size)(img_tensor)
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width

        if scale_method is not None:
            img_up = pil_img.resize((2 * pil_img.size[0], 2 * pil_img.size[1]), scale_method)
            img_up = to_tensor(img_up).unsqueeze(0)
            img_up = nn.ReplicationPad2d(self.pad_size * self.scale_factor)(img_up)

        patch_box = []
        # avoid the residual part is smaller than the padded size
        if height % self.seg_size < self.pad_size or width % self.seg_size < self.pad_size:
            self.seg_size += self.scale_factor * self.pad_size

        # split image into over-lapping pieces
        for i in range(self.pad_size, height, self.seg_size):
            for j in range(self.pad_size, width, self.seg_size):
                part = img_tensor[:, :,
                       (i - self.pad_size):min(i + self.pad_size + self.seg_size, height),
                       (j - self.pad_size):min(j + self.pad_size + self.seg_size, width)]
                if img_pad > 0:
                    part = nn.ZeroPad2d(img_pad)(part)
                if scale_method is not None:
                    # part_up = self.upsampler(part)
                    part_up = img_up[:, :,
                              self.scale_factor * (i - self.pad_size):min(i + self.pad_size + self.seg_size,
                                                                          height) * self.scale_factor,
                              self.scale_factor * (j - self.pad_size):min(j + self.pad_size + self.seg_size,
                                                                          width) * self.scale_factor]

                    patch_box.append((part, part_up))
                else:
                    patch_box.append(part)
        return patch_box

    def merge_img_tensor(self, list_img_tensor):
        out = torch.zeros((1, 3, self.height * self.scale_factor, self.width * self.scale_factor))
        img_tensors = copy.copy(list_img_tensor)
        rem = self.pad_size * 2

        pad_size = self.scale_factor * self.pad_size
        seg_size = self.scale_factor * self.seg_size
        height = self.scale_factor * self.height
        width = self.scale_factor * self.width
        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                part = img_tensors.pop(0)
                part = part[:, :, rem:-rem, rem:-rem]
                # might have error
                if len(part.size()) > 3:
                    _, _, p_h, p_w = part.size()
                    out[:, :, i:i + p_h, j:j + p_w] = part
                # out[:,:,
                # self.scale_factor*i:self.scale_factor*i+p_h,
                # self.scale_factor*j:self.scale_factor*j+p_w] = part
        out = out[:, :, rem:-rem, rem:-rem]
        return out


def load_single_image(img_file,
                      up_scale=False,
                      up_scale_factor=2,
                      up_scale_method=Image.BILINEAR,
                      zero_padding=False):
    img = Image.open(img_file).convert("RGB")
    out = to_tensor(img).unsqueeze(0)
    if zero_padding:
        out = nn.ZeroPad2d(zero_padding)(out)
    if up_scale:
        size = tuple(map(lambda x: x * up_scale_factor, img.size))
        img_up = img.resize(size, up_scale_method)
        img_up = to_tensor(img_up).unsqueeze(0)
        out = (out, img_up)

    return out


def standardize_img_format(img_folder):
    def process(img_file):
        img_path = os.path.dirname(img_file)
        img_name, _ = os.path.basename(img_file).split(".")
        out = os.path.join(img_path, img_name + ".JPEG")
        os.rename(img_file, out)

    list_imgs = []
    for i in ['png', "jpeg", 'jpg']:
        list_imgs.extend(glob.glob(img_folder + "**/*." + i, recursive=True))
    print("Found {} images.".format(len(list_imgs)))
    pool = ThreadPool(4)
    pool.map(process, list_imgs)
    pool.close()
    pool.join()
