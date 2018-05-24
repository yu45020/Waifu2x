from models import *
from utils import *
import time
from torchvision.utils import save_image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

img = convert_2_tensor('1.jpg')
img_pathes = ImagePatches(seg_size=12, upscale=2)
#
model = ESPCN_7(in_channels=3, upscale=2)
model = model.eval()



# start_t = time.time()
# out = model.forward_checkpoint(img)
# end_t = time.time()
# save_image(out, 'split2.jpg')
# print("Runtime :{}".format(end_t - start_t))

if __name__ == '__main__':
    img_pieces = img_pathes.split_img_tensor(img)
    start_t = time.time()
    out = []
    with Pool(cpu_count()) as p:
        out = p.map(model.forward_checkpoint, tqdm(img_pieces, ascii=True, unit='patch'))
        #
        # for pieces in tqdm(img_pieces, ascii=True):
        #     new = p.map(model.forward_checkpoint, pieces)
        #     out.append(new)
    end_t = time.time()
    bb = img_pathes.merge_imgs(out)
    save_image(bb, 'split2.jpg')
    print("Runtime :{}".format(end_t-start_t))

