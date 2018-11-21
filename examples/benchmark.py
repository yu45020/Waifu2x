import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import ImageData
from utils import image_quality
from utils.prepare_images import *

DCSCN_12 = "model_check_points/DCSCN/DCSCN_model_387epos_L12_noise_1.pt"
model_dcscn = torch.load(DCSCN_12)

model_upconv7 = UpConv_7()
model_upconv7.load_pre_train_weights("model_check_points/Upconv_7/anime/noise0_scale2.0x_model.json")

img_dataset = ImageData(img_folder='demo/demo_imgs/',
                        patch_size=96,
                        shrink_size=2,
                        noise_level=1,
                        down_sample_method=Image.BICUBIC,
                        color_mod='RGB')

img_data = DataLoader(img_dataset)  # DCSCN must set pad_img = 0
ssim_score = []
psnr_score = []
for img in tqdm(img_data, ascii=True):
    lr, hr = img
    out = model_upconv7.forward_checkpoint(lr)
    psnr_score.append(image_quality.calc_psnr(out, hr))
    ssim_score.append(image_quality.msssim(out, hr))

print("Averge PSNR score: {:.4f}".format(np.mean(psnr_score)))
print("Average MS-SSIM score: {:.4f}".format(np.mean(ssim_score)))
