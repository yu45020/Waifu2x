import time

from PIL import Image
from torchvision.utils import save_image

from models import *
from utils import *

# start_t = time.time()
# out = model.forward_checkpoint(img)
# end_t = time.time()
# save_image(out, 'split2.jpg')
# print("Runtime :{}".format(end_t - start_t))

if __name__ == '__main__':
    pre_trained = './model_results/ESPCN_7_14epos_model.pk'
    # rgb_weights = [0.29891*3, 0.58661*3, 0.11448*3]
    # rgb_weights = torch.FloatTensor(rgb_weights).view(3, 1, 1)
    img_pathes = ImagePatches(seg_size=600, upscale=2)
    model = ESPCN_7(in_channels=3, upscale=2)
    model.load_state_dict(torch.load(pre_trained, map_location='cpu'))
    model = model.eval()
    img = Image.open("2.jpg")
    img = to_tensor(img).unsqueeze(0)

    # img_pieces = img_pathes.split_img_tensor(img)
    start_t = time.time()
    # # cpu_count()
    # with Pool(1) as p:
    #     out = p.map(model.forward_checkpoint, tqdm(img_pieces, ascii=True, unit='patch'))
    #
    out = model.forward_checkpoint(img)
    end_t = time.time()
    # out = img_pathes.merge_imgs(out)
    # out = out*rgb_weights
    save_image(out, 'ESPCN_7.png', padding=0)
    print("Runtime :{}".format(end_t-start_t))


