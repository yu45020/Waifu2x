import numpy as np
from torch.optim import Adam
from tqdm import trange

from dataloader import *
from loss import *
from models import *
from utils import *

rgb_weights = [0.29891, 0.58661, 0.11448]
# https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

train_folder = './dataset/train/'
img_dataset = ImageData(train_folder,
                        max_patch_per_img=1000,
                        patch_size=1920,
                        shrink_size=2,
                        noise_level=1,
                        down_sample_method=Image.BICUBIC)

img_data = ImageLoader(img_dataset,
                       batch_size=10,
                       shuffle=True)

criteria = WeightedMSELoss(weights=rgb_weights)
model = ESPCN_7(in_channels=3, upscale=2)

learning_rate = 5e-4
weight_decay = 0
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
        lr_img, hr_img = batch
        model.zero_grad()

        outputs = model.forward(lr_img)
        loss = criteria(outputs, hr_img)
        loss.backward()
        optimizer.step()

        counter += 1
        all_loss.append(loss.item())
        batch_loss.append(loss.item())
        print("Current batch {} has loss {:.5f}".format(counter, loss.item()))

    one_ite_loss = np.mean(batch_loss)
    print("One iteration loss {:.5f}".format(one_ite_loss))


