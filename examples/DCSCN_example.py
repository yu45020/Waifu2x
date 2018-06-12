from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.measure import compare_psnr, compare_ssim
from torchvision.utils import save_image

from utils.prepare_images import *

pre_trained = "model_results/DCSCN_model_387epos.pt"
model = DCSCN(color_channel=3,
              up_scale=2,
              feature_layers=12,
              first_feature_filters=196,
              last_feature_filters=48,
              reconstruction_filters=128,
              up_sampler_filters=32)
model.load_state_dict(torch.load(pre_trained, map_location='cpu'))
lr = "./benchmark/miku_small.png"
img = Image.open(lr).convert("RGB")
img_up = img.resize((2 * img.size[0], 2 * img.size[1]), Image.BILINEAR)
img = to_tensor(img).unsqueeze(0)
img_up = to_tensor(img_up).unsqueeze(0)
out = model.forward_checkpoint((img, img_up))
save_image(out, './benchmark/miku_dcscn.png')

dcscn = imread("./benchmark/miku_dcscn.png")
hr = imread('./benchmark/miku_small_hq2x.png')
hr = rgba2rgb(hr)
waifu = imread("./benchmark/miku_small_waifu2x.png")

hr.shape
dcscn.shape

compare_psnr(hr, dcscn / 255)
compare_ssim(hr, dcscn / 255, multichannel=True)

compare_psnr(hr, waifu / 255)
compare_ssim(hr, waifu / 255, multichannel=True)

waifu = to_tensor(waifu).unsqueeze(0)
save_image(torch.cat([out, waifu]), './benchmark/compare.png')
