import time
from multiprocessing import Pool, cpu_count

from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from models import *
from utils import *

# start_t = time.time()
# out = model.forward_checkpoint(img)
# end_t = time.time()
# save_image(out, 'split2.jpg')
# print("Runtime :{}".format(end_t - start_t))

if __name__ == '__main__':
    pre_trained = './model_results/ESPCN_7_6epos_model.pk'
    rgb_weights = [0.29891, 0.58661, 0.11448]
    rgb_weights = torch.FloatTensor(rgb_weights).view(3, 1, 1)
    img_pathes = ImagePatches(seg_size=25, upscale=2)
    model = ESPCN_7(in_channels=3, upscale=2)
    model.load_state_dict(torch.load(pre_trained, map_location='cpu'))
    model = model.eval()
    img = Image.open("1.jpg")
    img = ToTensor()(img).unsqueeze(0)
    img_pieces = img_pathes.split_img_tensor(img)
    start_t = time.time()

    with Pool(cpu_count()) as p:
        out = p.map(model.forward_checkpoint, tqdm(img_pieces, ascii=True, unit='patch'))
        #
        # for pieces in tqdm(img_pieces, ascii=True):
        #     new = p.map(model.forward_checkpoint, pieces)
        #     out.append(new)
    end_t = time.time()
    out = img_pathes.merge_imgs(out)
    # out = out*rgb_weights
    save_image(out, 'ESPCN_7.png')
    print("Runtime :{}".format(end_t-start_t))

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor

img1 = Image.open("./dataset/train/003.png")
img1.mode

img1 = np.array(img1) / 255
img1 = torch.from_numpy(img1)
img1 = img1.transpose(0, 1).transpose(0, 2)

img2 = Image.open("ESPCN_7.png")
img2 = ToTensor()(img2)

img1.size()
np.sum(img1[0].numpy() - img2[0].numpy())
