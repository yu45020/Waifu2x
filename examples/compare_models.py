import time
from multiprocessing import Pool, cpu_count

from torchvision.utils import save_image

from utils.prepare_images import *

DCSCN_12 = "model_check_points/DCSCN/DCSCN_model_387epos_L12_noise_1.pt"
model_dcscn = torch.load(DCSCN_12)

model_upconv7 = UpConv_7()
model_upconv7.load_pre_train_weights("model_check_points/Upconv_7/anime/noise0_scale2.0x_model.json")

# demo_img = 'demo/demo_imgs/sp_twitter_icon_ao_3.png'
demo_img = "demo/demo_imgs/sp_twitter/sp_twitter_icon_ao_2.png"
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
    with Pool(cpu_count()) as p:
        out = p.map(model_dcscn.forward_checkpoint, img_patches)
    img_dcscn = img_splitter.merge_img_tensor(out)
    print(" DCSCN_12 model runtime: {:.3f}".format(time.time() - start_t))
    # 20s

    start_t = time.time()
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=model_upconv7.offset)
    with Pool(cpu_count()) as p:
        out = p.map(model_upconv7.forward_checkpoint, img_patches)
    img_upconv7 = img_splitter.merge_img_tensor(out)
    print("Upconv_7 runtime: {:.3f}".format(time.time() - start_t))
    # 5.3s
    final = torch.cat([img_t, img_bicubic, img_t, img_dcscn, img_t, img_upconv7])
    save_image(final, 'docs/demo_true_bicubic_dcscn_upconv.png', nrow=2)
