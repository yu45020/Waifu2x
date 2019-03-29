import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataloader import ImageData
from utils import image_quality
from utils.prepare_images import *
from Models import *

model_dcscn = DCSCN()
model_dcscn.load_state_dict(torch.load("model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt"))

model_upconv7 = UpConv_7()
model_upconv7.load_pre_train_weights("model_check_points/Upconv_7/noise1_scale2.0x_model.json")

model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
model_cran_v2.total_parameters()
model_cran_v2 = network_to_half(model_cran_v2)
model_cran_v2.load_state_dict(torch.load("model_check_points/CRAN_V2/CARN_model_checkpoint.pt"))
model_cran_v2 = model_cran_v2.float()

img_dataset = ImageData(img_folder='dataset/sp_twitter',
                        patch_size=96,
                        shrink_size=2,
                        noise_level=1,
                        down_sample_method=None,  # Image.BICUBIC,
                        color_mod='RGB')

img_data = DataLoader(img_dataset)  # DCSCN must set pad_img = 0
ssim_score = []
psnr_score = []

with torch.no_grad():
    for img in tqdm(img_data, ascii=True):
        lr, hr = img
        out = model_cran_v2(lr)
        psnr_score.append(image_quality.calc_psnr(out, hr))
        ssim_score.append(image_quality.msssim(out, hr))

print("Averge PSNR score: {:.4f}".format(np.mean(psnr_score)))
print("Average MS-SSIM score: {:.4f}".format(np.mean(ssim_score)))
