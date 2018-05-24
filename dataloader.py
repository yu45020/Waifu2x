import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor



class ImageData(Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.hr_folder = os.path.join(img_folder, 'hr')
        self.lr_folder = os.path.join(img_folder, 'lr')
        self.img_files = self.get_img_files()

    def get_img_files(self):
        file_names = os.listdir(self.hr_folder)
        basename = [os.path.basename(i).split(".")[0] for i in file_names]
        return basename

    @staticmethod
    def convert_2_tensor(img_file):
        img = Image.open(img_file)
        img = ToTensor()(img)
        return img

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        lr_img = os.path.join(self.lr_folder, img_file+".jpeg")
        lr_img = self.convert_2_tensor(lr_img)

        hr_img = os.path.join(self.hr_folder, img_file+".png")
        hr_img = self.convert_2_tensor(hr_img)
        return lr_img, hr_img

    @staticmethod
    def batch_collector(batch):
        lr_img = [i[0] for i in batch]
        lr_img = torch.stack(lr_img, dim=0)

        hr_img = [i[1] for i in batch]
        hr_img = torch.stack(hr_img, dim=0)
        if use_cuda:
            lr_img = lr_img.cuda()
            hr_img.cuda(async=True)
        return lr_img, hr_img


class ImageLoader(DataLoader):
    def __init__(self, img_folder, batch_size=10, shuffle=True):
        self.img_dataset = ImageData(img_folder)
        super(ImageLoader, self).__init__(self.img_dataset, batch_size, shuffle,
                                          collate_fn=self.img_dataset.batch_collector)


if __name__ == '__main__':
    train_folder = './dataset/train/'
    img_data = ImageLoader(train_folder)
    for i in img_data:
        a = i
        break

    lr = a[0]
    lr.size()
    hr = a[1]
    hr.size()
