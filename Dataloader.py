import glob
import io
import numpy as np
import re
import os
import random
from io import BytesIO
from uuid import uuid4
import sqlite3
import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor


class ImageH5Data(Dataset):
    def __init__(self, h5py_file, folder_name):
        self.data = h5py.File(h5py_file, 'r')[folder_name]
        self.data_hr = self.data['train_hr']
        self.data_lr = self.data['train_lr']
        self.len_imgs = len(self.data_hr)
        self.h5py_file = h5py_file
        self.folder_name = folder_name

    def __len__(self):
        # with h5py.File(self.h5py_file, 'r') as f:
        #     return len(f[self.folder_name]['train_lr'])
        return self.len_imgs

    def __getitem__(self, index):
        # with h5py.File(self.h5py_file, 'r') as f:
        #     data_lr = f[self.folder_name]['train_lr'][index]
        #     data_hr = f[self.folder_name]['train_lr'][index]
        #
        #     return data_lr, data_hr
        return self.data_lr[index], self.data_hr[index]


class ImageData(Dataset):
    def __init__(self,
                 img_folder,
                 patch_size=96,
                 shrink_size=2,
                 noise_level=1,
                 down_sample_method=None,
                 color_mod='RGB',
                 dummy_len=None):

        self.img_folder = img_folder
        all_img = glob.glob(self.img_folder + "/**", recursive=True)
        self.img = list(filter(lambda x: x.endswith('png') or x.endswith("jpg") or x.endswith("jpeg"), all_img))
        self.total_img = len(self.img)
        self.dummy_len = dummy_len if dummy_len is not None else self.total_img
        self.random_cropper = RandomCrop(size=patch_size)
        self.color_mod = color_mod
        self.img_augmenter = ImageAugment(shrink_size, noise_level, down_sample_method)

    def get_img_patches(self, img_file):
        img_pil = Image.open(img_file).convert("RGB")
        img_patch = self.random_cropper(img_pil)
        lr_hr_patches = self.img_augmenter.process(img_patch)
        return lr_hr_patches

    def __len__(self):
        return self.dummy_len  # len(self.img)

    def __getitem__(self, index):
        idx = random.choice(range(0, self.total_img))
        img = self.img[idx]
        patch = self.get_img_patches(img)
        if self.color_mod == 'RGB':
            lr_img = patch[0].convert("RGB")
            hr_img = patch[1].convert("RGB")
        elif self.color_mod == 'YCbCr':
            lr_img, _, _ = patch[0].convert('YCbCr').split()
            hr_img, _, _ = patch[1].convert('YCbCr').split()
        else:
            raise KeyError('Either RGB or YCbCr')
        return to_tensor(lr_img), to_tensor(hr_img)


class Image2Sqlite(ImageData):
    def __getitem__(self, item):
        img = self.img[item]
        lr_hr_patch = self.get_img_patches(img)
        if self.color_mod == 'RGB':
            lr_img = lr_hr_patch[0].convert("RGB")
            hr_img = lr_hr_patch[1].convert("RGB")
        elif self.color_mod == 'YCbCr':
            lr_img, _, _ = lr_hr_patch[0].convert('YCbCr').split()
            hr_img, _, _ = lr_hr_patch[1].convert('YCbCr').split()
        else:
            raise KeyError('Either RGB or YCbCr')
        lr_byte = self.convert_to_bytevalue(lr_img)
        hr_byte = self.convert_to_bytevalue(hr_img)
        return [lr_byte, hr_byte]

    @staticmethod
    def convert_to_bytevalue(pil_img):
        img_byte = io.BytesIO()
        pil_img.save(img_byte, format='png')
        return img_byte.getvalue()


class ImageDBData(Dataset):
    def __init__(self, db_file, db_table="images", lr_col="lr_img", hr_col="hr_img", max_images=None):
        self.db_file = db_file
        self.db_table = db_table
        self.lr_col = lr_col
        self.hr_col = hr_col
        self.total_images = self.get_num_rows(max_images)
        # self.lr_hr_images = self.get_all_images()

    def __len__(self):
        return self.total_images

    # def get_all_images(self):
    #     with sqlite3.connect(self.db_file) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute(f"SELECT * FROM {self.db_table} LIMIT {self.total_images}")
    #         return cursor.fetchall()

    def get_num_rows(self, max_images):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT MAX(ROWID) FROM {self.db_table}")
            db_rows = cursor.fetchone()[0]
        if max_images:
            return min(max_images, db_rows)
        else:
            return db_rows

    def __getitem__(self, item):
        # lr, hr = self.lr_hr_images[item]
        # lr = Image.open(io.BytesIO(lr))
        # hr = Image.open(io.BytesIO(hr))
        # return to_tensor(lr), to_tensor(hr)
        # note sqlite rowid starts with 1
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT {self.lr_col}, {self.hr_col} FROM {self.db_table} WHERE ROWID={item + 1}")
            lr, hr = cursor.fetchone()
            lr = Image.open(io.BytesIO(lr)).convert("RGB")
            hr = Image.open(io.BytesIO(hr)).convert("RGB")
            # lr = np.array(lr)  # use scale [0, 255] instead of [0,1]
            # hr = np.array(hr)
            return to_tensor(lr), to_tensor(hr)


class ImagePatchData(Dataset):
    def __init__(self, lr_folder, hr_folder):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.lr_imgs = glob.glob(os.path.join(lr_folder, "**"))
        self.total_imgs = len(self.lr_imgs)

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, item):
        lr_file = self.lr_imgs[item]
        hr_path = re.sub("lr", 'hr', os.path.dirname(lr_file))
        filename = os.path.basename(lr_file)
        hr_file = os.path.join(hr_path, filename)
        return to_tensor(Image.open(lr_file)), to_tensor(Image.open(hr_file))


class ImageAugment:
    def __init__(self,
                 shrink_size=2,
                 noise_level=1,
                 down_sample_method=None
                 ):
        # noise_level (int): 0: no noise; 1: 75-95% quality; 2:50-75%
        if noise_level == 0:
            self.noise_level = [0, 0]
        elif noise_level == 1:
            self.noise_level = [5, 25]
        elif noise_level == 2:
            self.noise_level = [25, 50]
        else:
            raise KeyError("Noise level should be either 0, 1, 2")
        self.shrink_size = shrink_size
        self.down_sample_method = down_sample_method

    def shrink_img(self, hr_img):

        if self.down_sample_method is None:
            resample_method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        else:
            resample_method = self.down_sample_method
        img_w, img_h = tuple(map(lambda x: int(x / self.shrink_size), hr_img.size))
        lr_img = hr_img.resize((img_w, img_h), resample_method)
        return lr_img

    def add_jpeg_noise(self, hr_img):
        quality = 100 - round(random.uniform(*self.noise_level))
        lr_img = BytesIO()
        hr_img.save(lr_img, format='JPEG', quality=quality)
        lr_img.seek(0)
        lr_img = Image.open(lr_img)
        return lr_img

    def process(self, hr_patch_pil):
        lr_patch_pil = self.shrink_img(hr_patch_pil)
        if self.noise_level[1] > 0:
            lr_patch_pil = self.add_jpeg_noise(lr_patch_pil)

        return lr_patch_pil, hr_patch_pil

    def up_sample(self, img, resample):
        width, height = img.size
        return img.resize((self.shrink_size * width, self.shrink_size * height), resample=resample)
