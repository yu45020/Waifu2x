import random
import time
from io import BytesIO
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import glob
from itertools import chain
from torchvision.transforms.functional import to_tensor
from multiprocessing import Pool, cpu_count
from torchvision.utils import save_image
from multiprocessing.dummy import Pool as ThreadPool

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class ImageData(Dataset):
    def __init__(self,
                 img_folder,
                 max_patch_per_img,
                 patch_size,
                 shrink_size,
                 noise_level,
                 down_sample_method):
        self.img_folder = img_folder
        self.max_path_per_img = max_patch_per_img
        self.patch_size = patch_size

        self.patch_grids = self.get_img_patch_grids()

        self.img_augmenter = ImageAugment(shrink_size, noise_level, down_sample_method)

    def get_img_patch_grids(self):
        file_names = glob.glob(self.img_folder + '**/*.png', recursive=True)
        patch_grids = list(map(self.get_img_grids, file_names))
        patch_grids = list(chain.from_iterable(patch_grids))
        print("Find {} patches from {} images.".format(len(patch_grids), len(file_names)))
        return patch_grids

    def get_img_grids(self, img_file):
        # return nested list [ [img_file, position tuple], ...]
        img = Image.open(img_file)
        img_w, img_h = img.size
        count_w, count_h = img_w // self.patch_size, img_h // self.patch_size

        # (left, top, right button) square's 4 lines as coordinate
        patch_box = [
            [img_file, (self.patch_size * i, self.patch_size * j, self.patch_size * (i + 1), self.patch_size * (j + 1))]
            for i in range(count_w) for j in range(count_h)
        ]

        if len(patch_box) > self.max_path_per_img:
            patch_box = random.sample(patch_box, self.max_path_per_img)

        return patch_box

    def __len__(self):
        return len(self.patch_grids)

    def __getitem__(self, index):
        patch = self.patch_grids[index]
        return patch
        # lr_patch, hr_patch = self.img_augmenter.process(patch)
        # return to_tensor(lr_patch), to_tensor(hr_patch)


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
        if self.noise_level > 0:
            lr_patch = self.add_jpeg_noise(lr_patch)
        return lr_patch, hr_patch


class ImageLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        # self.img_dataset = ImageData(img_folder,
        #                              max_patch_per_img,
        #                              patch_size,
        #                              shrink_size,
        #                              noise_level,
        #                              down_sample_method)
        self.dataset = dataset
        super(ImageLoader, self).__init__(dataset, batch_size, shuffle,
                                          collate_fn=self.batch_collector,
                                          num_workers=num_workers)

    # def merge_batch(self, batch):
    #     unique_files = set([i[0] for i in batch])
    #     file_dict = {i: [] for i in unique_files}
    #     for i in batch:
    #         file_dict[i[0]].append(i[1])
    #
    #     lr_hr_patches = []
    #     for file in file_dict:
    #         img = Image.open(file)
    #         grids = file_dict[file]
    #         lr_hr_patches += [self.dataset.img_augmenter.process(img, i) for i in grids]
    #     return lr_hr_patches

    def merge_batch(self, batch):
        unique_files = set([i[0] for i in batch])
        self.file_dict = {i: [] for i in unique_files}
        for i in batch:
            self.file_dict[i[0]].append(i[1])

        pool = ThreadPool(4)
        lr_hr_patches = pool.map(self.crop_img, unique_files)
        pool.close()
        pool.join()

        return lr_hr_patches

    def crop_img(self, file):
        img = Image.open(file)
        grids = self.file_dict[file]
        return [self.dataset.img_augmenter.process(img, i) for i in grids]

    def batch_collector(self, batch):
        lr_hr_patch = self.merge_batch(batch)
        lr_hr_patch = list(chain.from_iterable(lr_hr_patch))
        lr_img = [to_tensor(i[0]) for i in lr_hr_patch]
        lr_img = torch.stack(lr_img, dim=0).contiguous()

        hr_img = [to_tensor(i[1]) for i in lr_hr_patch]
        hr_img = torch.stack(hr_img, dim=0).contiguous()
        if use_cuda:
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda(async=True)
        return lr_img, hr_img


if __name__ == '__main__':
    train_folder = './dataset/train/'
    img_dataset = ImageData(img_folder=train_folder,
                            max_patch_per_img=1000,
                            patch_size=600,
                            shrink_size=2,
                            noise_level=1,
                            down_sample_method=Image.BICUBIC)

    img_data = ImageLoader(img_dataset, batch_size=10, shuffle=True)

    b = iter(img_data)
    a = next(b)
    lr = a[0]
    print(lr.size())
    hr = a[1]
    print(hr.size())
