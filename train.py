from models import *
from utils import *
import time
from torchvision.utils import save_image
from tqdm import tqdm
from multiprocessing import Process, Pool, cpu_count

img = convert_2_tensor('1.png')
img_pathes = ImagePatches(seg_size=100, upscale=2)

model = ESPCN_5(in_channels=3, upscale=2)
model.eval()


if __name__ == '__main__':
    img_pieces = img_pathes.split_img_tensor(img)
    start_t = time.time()
    out = []
    with Pool(cpu_count()) as p:
        for pieces in tqdm(img_pieces, ascii=True):
            new = p.map(model.forward_checkpoint, pieces)
            out.append(new)

    end_t = time.time()
    bb = img_pathes.merge_imgs(out)
    save_image(bb, 'split2.jpg')
    print("Runtime :{}".format(end_t-start_t))
