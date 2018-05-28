import time
from multiprocessing import cpu_count, Pool

from torchvision.utils import save_image
from tqdm import tqdm

from models import *
from utils import *

if __name__ == '__main__':
    pre_trained = './model_results/DCSCN_model_55epos.pt'
    model = DCSCN(color_channel=3,
                  up_scale=2,
                  feature_layers=12,
                  first_feature_filters=196,
                  last_feature_filters=48,
                  reconstruction_filters=64,
                  up_sampler_filters=32,
                  dropout_rate=0)

    model.load_state_dict(torch.load(pre_trained, map_location='cpu'))
    model = model.eval()
    model = Check()
    img_splitter = ImagePatches(seg_size=17, upscale=2)
    img = Image.open("2_half.jpeg").convert("RGB")
    img_pieces = img_splitter.split_img_tensor(img)

    start_t = time.time()
    with Pool(cpu_count()) as p:
        out = p.map(model.forward_checkpoint, tqdm(img_pieces, ascii=True, unit='patch'))

    out = img_splitter.merge_imgs(out)

    img_up = img.resize((2 * img.size[0], 2 * img.size[1]), Image.BILINEAR)

    img_up = to_tensor(img_up).unsqueeze(0)
    final = out + img_up
    save_image(out, 'residual.png', padding=0)
    save_image(final, 'final.png', padding=0)
    end_t = time.time()
    print("Runtime :{}".format(end_t - start_t))

#
# save_image(out, 'out.png')
# img = Image.open('residual.png')
# from torchvision.transforms.functional import adjust_saturation, adjust_brightness,adjust_contrast
# img_2 = adjust_contrast(img, 10)
# img_2.save("residual_2.png")
