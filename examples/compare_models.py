import time
from multiprocessing import Pool, cpu_count

from torchvision.utils import save_image
from Models import *
from utils.prepare_images import *

model_dcscn = DCSCN()
model_dcscn.load_state_dict(torch.load("model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt"))

model_upconv7 = UpConv_7()
model_upconv7.load_pre_train_weights("model_check_points/Upconv_7/noise1_scale2.0x_model.json")

model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
model_cran_v2 = network_to_half(model_cran_v2)
model_cran_v2.load_state_dict(torch.load("model_check_points/CRAN_V2/CARN_model_checkpoint.pt", 'cpu'))
model_cran_v2 = model_cran_v2.float()
# demo_img = 'demo/demo_imgs/sp_twitter_icon_ao_3.png'
demo_img = "dataset/sp_twitter/sp_twitter_icon_ao_3.png"
img = Image.open(demo_img).convert("RGB")
print("Demo image size : {}".format(img.size))
img_t = to_tensor(img).unsqueeze(0)

img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC)
img_bicubic = img.resize((img.size[0] * 2, img.size[1] * 2), Image.BICUBIC)
img_bicubic = to_tensor(img_bicubic).unsqueeze(0)

if __name__ == '__main__':
    img_splitter = ImageSplitter(seg_size=48, scale_factor=2, boarder_pad_size=3)

    print("Runing DCSCN models...")
    start_t = time.time()
    img_patches = img_splitter.split_img_tensor(img, scale_method=Image.BILINEAR, img_pad=0)
    with torch.no_grad():
        out = [model_dcscn(i) for i in img_patches]
    # with Pool(cpu_count()) as p:
    #     out = p.map(model_dcscn.forward_checkpoint, img_patches)
    img_dcscn = img_splitter.merge_img_tensor(out)
    print(" DCSCN_12 model runtime: {:.3f}".format(time.time() - start_t))

    # start_t = time.time()
    # img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=model_upconv7.offset)
    # # with Pool(cpu_count()) as p:
    # #     out = p.map(model_upconv7.forward_checkpoint, img_patches)
    # with torch.no_grad():
    #     out = [model_upconv7(i) for i in img_patches]
    #     [i.size() for i in img_patches]
    # img_upconv7 = img_splitter.merge_img_tensor(out)
    # print("Upconv_7 runtime: {:.3f}".format(time.time() - start_t))
    #

    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    start_t = time.time()
    with torch.no_grad():
        out = [model_cran_v2(i) for i in img_patches]
    img_cran_v2 = img_splitter.merge_img_tensor(out)
    print("CRAN V2 runtime: {:.3f}".format(time.time() - start_t))

    final = torch.cat([img_t, img_bicubic,
                       img_t, img_dcscn,
                       img_t, img_cran_v2])
    save_image(final, 'docs/demo_bicubic_model_comparison.png', nrow=2)
