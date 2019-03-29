import glob

import h5py
from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from Dataloader import ImageAugment

patch_size = 128
shrink_size = 2
noise_level = 1
patches_per_img = 20
images = glob.glob("dataset/train/*")

database = h5py.File("train_images.hdf5", 'w')

dat_group = database.create_group("shrink_2_noise_level_1_downsample_random_rgb")
# del database['shrink_2_noise_level_1_downsample_random']
storage_lr = dat_group.create_dataset("train_lr", shape=(patches_per_img * len(images), 3,
                                                         patch_size // shrink_size,
                                                         patch_size // shrink_size),
                                      dtype='float32',
                                      # compression='lzf',
                                      )
storage_hr = dat_group.create_dataset("train_hr", shape=(patches_per_img * len(images), 3,
                                                         patch_size, patch_size),
                                      # compression='lzf',
                                      dtype='float32')

random_cropper = RandomCrop(size=patch_size)
img_augmenter = ImageAugment(shrink_size, noise_level, down_sample_method=None)


def get_img_patches(img_pil):
    img_patch = random_cropper(img_pil)
    lr_hr_patches = img_augmenter.process(img_patch)
    return lr_hr_patches


counter = 0
for img in tqdm(images):
    img_pil = Image.open(img).convert("RGB")
    for i in range(patches_per_img):
        patch = get_img_patches(img_pil)
        storage_lr[counter] = to_tensor(patch[0].convert("RGB")).numpy()
        storage_hr[counter] = to_tensor(patch[1].convert("RGB")).numpy()
        counter += 1
database.close()
