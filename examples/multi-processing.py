# Speed up inference in CPU only environment

import time
from multiprocessing import Pool, cpu_count

from torchvision.utils import save_image

from utils.prepare_images import *

# pre_train = './model_results/DCSCN_model_387epos.pt'
# model = DCSCN(color_channel=3,
#               up_scale=2,
#               feature_layers=12,
#               first_feature_filters=196,
#               last_feature_filters=48,
#               reconstruction_filters=128,
#               up_sampler_filters=32)
# model.load_state_dict(torch.load(pre_train, map_location='cpu'))
model = UpConv_7()
model.load_pre_train_weights('./model_check_points/Upconv_7/noise1_scale2.0x_model.json')

lr_imgs = glob.glob("./benchmark/Set14_SR/lr_imgs/*_LR.png")

if __name__ == '__main__':
    # model = torch.load("dcscn.pt")  # set the img_pad=0
    sp = ImageSplitter(seg_size=48, scale_factor=2, boarder_pad_size=3)
    for lr_img in lr_imgs:
        a = time.time()
        patches = sp.split_img_tensor(Image.open(lr_img))
        with Pool(cpu_count()) as p:
            out = p.map(model.forward_checkpoint, patches)
        final = sp.merge_img_tensor(out)
        save_image(final, lr_img + "waifu.png", padding=0)
        print(time.time() - a)
#
