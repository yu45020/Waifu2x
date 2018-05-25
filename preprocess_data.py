import shutil

import PIL
import glob
from PIL import Image
from multiprocessing import Pool, cpu_count
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
                 patch_size=48
                 ):
        # noise_level (int): 0: no noise; 1: 90% quality; 2:80%
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        if not os.path.isdir(out_folder + "hr/"):
            os.mkdir(out_folder + 'hr/')

        if not os.path.isdir(temp_folder):
            os.mkdir(temp_folder)

        self.temp_folder = temp_folder

        self.out_folder = out_folder
        self.max_path_per_img = max_patch_per_img
        self.patch_size = patch_size

        self.img_cache = []
        self.folder_count = 0

    def none_overlap_patch_crop(self, img):
        img_patches = self.get_img_grids(img, self.patch_size)
        if len(img_patches) > self.max_path_per_img:
            img_patches = random.sample(img_patches, self.max_path_per_img)
        return [img.crop(i) for i in img_patches]

    @staticmethod
    def get_img_grids(img, patch_size):
        img_w, img_h = img.size
        count_w, count_h = img_w // patch_size, img_h // patch_size
        patch_box = [(patch_size * i, patch_size * j, patch_size * (i + 1), patch_size * (j + 1))
                     for i in range(count_w) for j in range(count_h)]
        return patch_box

    def process(self, img_file):
        img = Image.open(img_file)
        hr_patch = self.none_overlap_patch_crop(img)
        file_names = [self.save(i) for i in hr_patch]
        return file_names

    def save(self, hr_img):
        name = uuid.uuid4()
        file_name = self.out_folder + "hr/{}.png".format(name)
        hr_img.save(file_name, format='PNG')
        return file_name

    def split_img_2_folders(self, file_names):
        self.img_cache += file_names
        if len(self.img_cache) >= 1000:
            new_folder = os.path.join(self.out_folder, "hr", str(self.folder_count))
            os.mkdir(new_folder)
            [shutil.move(i, new_folder) for i in self.img_cache]
            self.img_cache = []
            self.folder_count += 1

all_images = glob.glob(img_source)

augmenter = ImageAugment(out_folder=train_folder,
                         temp_folder=temp_folder,
                         max_patch_per_img=1000,
                         patch_size=1200,  # high resolution images
                         )
print("Find {} images.".format(len(all_images)))
list(map(augmenter.process, all_images))

if __name__ == "__main__":
    all_images = glob.glob(img_source)
    augmenter = ImageAugment(out_folder=train_folder,
                             temp_folder=temp_folder,
                             max_patch_per_img=1000,
                             patch_size=192,  # high resolution images
                             )
    print("Find {} images.".format(len(all_images)))
    with Pool(cpu_count()) as p:
        out = p.map(augmenter.process, all_images)
