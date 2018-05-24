import PIL
import glob
from PIL import Image
from multiprocessing import Pool, cpu_count
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import Augmentor
import random
import uuid
import os
from tqdm import tqdm

train_folder = './dataset/train/'
img_source = "./dataset/raw/**/*.*"
temp_folder = './dataset/temp/'


# img.size  # horizontal, vertical

class ImageAugment:
    def __init__(self, out_folder,
                 temp_folder,
                 max_patch_per_img=100,
                 patch_size=48,
                 shrink_size=2,
                 noise_level=0
                 ):
        # noise_level (int): 0: no noise; 1: 90% quality; 2:80%
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        if not os.path.isdir(out_folder + "hr/"):
            os.mkdir(out_folder + 'hr/')
            os.mkdir(out_folder + 'lr/')

        if not os.path.isdir(temp_folder):
            os.mkdir(temp_folder)
        self.temp_folder = temp_folder

        self.out_folder = out_folder
        self.max_path_per_img = max_patch_per_img
        self.shrink_size = shrink_size
        self.noise_level = noise_level
        self.patch_size = patch_size

    def none_overlap_patch_crop(self, img):
        img_patches = self.get_img_grids(img, self.patch_size)
        if len(img_patches) > self.max_path_per_img:
            img_patches = random.sample(img_patches, self.max_path_per_img)
        return [img.crop(i) for i in img_patches]

    @staticmethod
    def get_img_grids(img, patch_size):
        # output a list of tuples
        # the box starts from upper left and moves vertically, so the right corner may be dropped
        img_w, img_h = img.size
        patch_box = []
        for w in range(0, img_w, patch_size):
            for h in range(0, img_h, patch_size):
                if (w + patch_size) <= img_h and (h + patch_size) <= img_w:
                    box = (h, w, h + patch_size, w + patch_size)
                    patch_box.append(box)

        return patch_box

    def shrink_image(self, img_hr):
        size = tuple(map(lambda x: int(x / self.shrink_size), img_hr.size))
        img_lr = img_hr.resize(size, resample=PIL.Image.NEAREST)
        return img_lr, img_hr

    def add_noise(self, img_tuple):
        quality = 100 - 10 * self.noise_level
        img_lr, img_hr = img_tuple
        temp = self.temp_folder + str(uuid.uuid4()) + '.jpeg'
        img_lr.save(temp, format='JPEG', quality=quality)
        img_lr = Image.open(temp)
        return img_lr, img_hr

    def process(self, img_file):
        img = Image.open(img_file)
        hr_patch = self.none_overlap_patch_crop(img)
        lr_hr_patch = [self.shrink_image(i) for i in hr_patch]
        if self.noise_level > 0:
            lr_hr_patch = [self.add_noise(i) for i in lr_hr_patch]
        # tensor_patch = [self.convert_2_tensor(i) for i in lr_hr_patch]
        [self.save(i) for i in lr_hr_patch]

    # def convert_2_tensor(self, lr_hr_patch):
    #     lr_patch, hr_patch = lr_hr_patch
    #     lr_patch = ToTensor()(lr_patch)
    #     hr_patch = ToTensor()(hr_patch)
    #     return {'lr': lr_patch, 'hr': hr_patch}

    def clean_temp(self):
        if self.noise_level > 0:
            all_temps = glob.glob(self.temp_folder + '*.*')
            [os.remove(file) for file in all_temps]

    def save(self, img_patch):
        lr_img, hr_img = img_patch
        file_name = uuid.uuid4()
        lr_img.save(self.out_folder + 'lr/{}.jpeg'.format(file_name), format='JPEG', quality=100)
        hr_img.save(self.out_folder + "hr/{}.png".format(file_name), format='PNG')


if __name__ == '__main__':
    all_images = glob.glob(img_source, recursive=True)
    augmenter = ImageAugment(out_folder=train_folder,
                             temp_folder=temp_folder,
                             patch_size=192,  # high resolution images
                             shrink_size=2,  # low resolution images
                             noise_level=1,
                             max_patch_per_img=1000
                             )
    print("Find {} images.".format(len(all_images)))
    with Pool(cpu_count()) as p:
        p.map(augmenter.process, tqdm(all_images, ascii=True))

    # augmenter.save(out)
    augmenter.clean_temp()



from shutil import make_archive

make_archive('./dataset/train_data_hr.zip', format='zip',
             root_dir="./dataset/train/hr/")
make_archive('./dataset/train_data_lr.zip', format='zip',
             root_dir='./dataset/train/lr/')