import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import trange

from dataloader import *
from models import *
from utils import *

# rgb_weights = [0.29891 * 3, 0.58661 * 3, 0.11448 * 3]
# https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109


# http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

train_folder = './dataset/train/'
test_folder = "./dataset/test/"

img_dataset = ImageData(train_folder,
                        max_patch_per_img=1000,
                        patch_size=96,
                        shrink_size=2,
                        noise_level=1,
                        down_sample_method=Image.BILINEAR,
                        up_sample_method=Image.BILINEAR,
                        color_mod='RGB')

img_data = ImageLoader(img_dataset,
                       up_sample=True,
                       batch_size=25,
                       shuffle=True)
total_batch = len(img_data)
print(total_batch)
test_dataset = ImageData(test_folder,
                         max_patch_per_img=10,
                         patch_size=96,
                         shrink_size=2,
                         noise_level=1,
                         down_sample_method=Image.BILINEAR,
                         up_sample_method=Image.BILINEAR,
                         color_mod='RGB')

test_data = ImageLoader(test_dataset,
                        up_sample=True,
                        batch_size=10,
                        shuffle=False)
# criteria = WeightedHuberLoss(weights=rgb_weights)
criteria = nn.L1Loss()
model = DCSCN(color_channel=3,
              up_scale=2,
              feature_layers=12,
              first_feature_filters=196,
              last_feature_filters=48,
              reconstruction_filters=64,
              up_sampler_filters=32)

pre_train = torch.load("./model_results/DCSCN_model_135epos.pt", map_location='cpu')
new = model.state_dict()
a = {}
for i, j in zip(pre_train, new):
    a[j] = pre_train[i]

model.load_state_dict(a, strict=False)
torch.save(model.state_dict(), "./model_results/DCSCN_model_135epos_2.pt")

learning_rate = 5e-3
weight_decay = 0
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

if use_cuda:
    model = model.cuda()
    criteria = criteria.cuda()

iteration = 1
all_loss = []
avg_ssim = []
counter = 0

ibar = trange(iteration, ascii=True, maxinterval=1,
              postfix={"avg_loss": 0, "train_ssim": 0, "test_ssim": 0})

for i in ibar:
    batch_loss = []
    insample_ssim = 0
    for index, batch in enumerate(img_data):
        lr_img, hr_img = batch
        model.zero_grad()

        outputs = model.forward(lr_img)
        loss = criteria(outputs, hr_img)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        counter += 1
        all_loss.append(loss.item())
        batch_loss.append(loss.item())
        ssim = eval_ssim(outputs, hr_img)
        ibar.set_postfix(ratio=index / total_batch, loss=loss.item(), ssim=ssim, batch=index)
        insample_ssim += ssim

    one_ite_loss = np.mean(batch_loss)
    print("One iteration loss {:.5f}".format(one_ite_loss))

    # eval
    ssim = 0
    for test_batch in test_data:
        lr_img, hr_img
        lr_img_up = model.forward_checkpoint(lr_img)
        ssim += eval_ssim(lr_img_up, hr_img)
    avg_ssim.append(ssim / len(test_data))
    ibar.write("Average loss {:.5f}; train SSIM {:.5f}; test SSIM {:.5f}".format(one_ite_loss,
                                                                                 insample_ssim / total_batch,
                                                                                 ssim / len(test_data)))

torch.save(model.state_dict(), 'DCSCN_10epos.pt')
torch.save(optimizer.state_dict(), 'DCSCN_optim_10epos.pt')

import numpy as np
from PIL import Image

img = Image.open('2.png')
img = img.resize((3, 3))
np.array(img)[1]
img2 = img.resize((6, 6), Image.BILINEAR)
np.array(img2)[1]
a = next(iter(model.parameters()))
a.grad
clip_grad_norm_
