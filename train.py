import numpy as np
from torch.optim import Adam
from tqdm import trange

from dataloader import *
from loss import WeightedHuberLoss
from models import *
from utils import *

rgb_weights = [0.29891 * 3, 0.58661 * 3, 0.11448 * 3]
# https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

train_folder = './dataset/train/'
img_dataset = ImageData(train_folder,
                        max_patch_per_img=1000,
                        patch_size=96,
                        shrink_size=2,
                        noise_level=1,
                        down_sample_method=Image.BICUBIC,
                        up_sample_method=Image.LANCZOS)

img_data = ImageLoader(img_dataset,
                       up_sample=True,
                       batch_size=20,
                       shuffle=True)

criteria = WeightedHuberLoss(weights=rgb_weights)
model = DCSCN(color_channel=3,
              up_scale=2,
              feature_layers=12,
              first_feature_filters=96,
              last_feature_filters=48,
              reconstruction_filters=64,
              up_sampler_filters=32)

learning_rate = 5e-4
weight_decay = 1e-4
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

if use_cuda:
    model = model.cuda()
    criteria = criteria.cuda()

iteration = 1
all_loss = []
counter = 0

for i in trange(iteration, ascii=True):
    batch_loss = []
    for batch in img_data:
        lr_img, lr_img_up, hr_img = batch
        model.zero_grad()

        outputs = model.forward(lr_img)
        outputs += lr_img_up
        loss = criteria(outputs, hr_img)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        counter += 1
        all_loss.append(loss.item())
        batch_loss.append(loss.item())
        print("Current batch {} has loss {:.5f}".format(counter, loss.item()))

    one_ite_loss = np.mean(batch_loss)
    print("One iteration loss {:.5f}".format(one_ite_loss))


torch.save(optimizer.state_dict())
model.modules()
