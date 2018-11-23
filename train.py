import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from dataloader import *
from utils import image_quality
from utils.prepare_images import *

# rgb_weights = [0.29891 * 3, 0.58661 * 3, 0.11448 * 3]
# https://github.com/nagadomi/waifu2x/blob/master/train.lua#L109


# http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

train_folder = './dataset/train'
test_folder = "./dataset/test"

img_dataset = ImageData(train_folder,
                        patch_size=96,
                        shrink_size=2,
                        noise_level=1,
                        down_sample_method=None,
                        color_mod='RGB')

img_data = DataLoader(img_dataset, batch_size=2, shuffle=True)

total_batch = len(img_data)
print(total_batch)

test_dataset = ImageData(test_folder,
                         patch_size=96,
                         shrink_size=2,
                         noise_level=1,
                         down_sample_method=Image.BICUBIC,
                         color_mod='RGB')
num_test = len(test_dataset)
test_data = DataLoader(test_dataset)

# criteria = WeightedHuberLoss(weights=rgb_weights)
criteria = nn.L1Loss()

model = DCSCN(color_channel=3,
              up_scale=2,
              feature_layers=12,
              first_feature_filters=196,
              last_feature_filters=48,
              reconstruction_filters=128,
              up_sampler_filters=32)

pre_train = torch.load("./model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt", map_location='cpu')
# new = model.state_dict()
# a = {}
# for i, j in zip(pre_train, new):
#     a[j] = pre_train[i]
#
model.load_state_dict(pre_train, strict=False)

# torch.save(model.state_dict(), "./model_results/DCSCN_model_135epos_2.pt")

learning_rate = 5e-3
weight_decay = 4e-5
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

model = model.cuda()
# criteria = criteria.cuda()

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
        lr_img = lr_img.cuda()
        hr_img = hr_img.cuda()
        lr_up = nn.functional.interpolate(lr_img, scale_factor=2,
                                          mode='bicubic', align_corners=False)
        model.zero_grad()
        outputs = model.forward((lr_img, lr_up))
        loss = criteria(outputs, hr_img)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        counter += 1
        all_loss.append(loss.item())
        batch_loss.append(loss.item())
        ssim = image_quality.msssim(outputs, hr_img)
        ibar.set_postfix(ratio=index / total_batch, loss=loss.item(), ssim=ssim, batch=index)
        insample_ssim += ssim

    one_ite_loss = np.mean(batch_loss)
    print("One iteration loss {:.5f}".format(one_ite_loss))

    # eval
    ssim = 0
    for test_batch in test_data:
        lr_img, hr_img = test_batch
        lr_img_up = model.forward_checkpoint(lr_img)
        ssim += image_quality.msssim(lr_img_up, hr_img)
    avg_ssim.append(ssim / num_test)
    ibar.write("Average loss {:.5f}; train SSIM {:.5f}; test SSIM {:.5f}".format(one_ite_loss,
                                                                                 insample_ssim / total_batch,
                                                                                 ssim / len(test_data)))

torch.save(model.state_dict(), './model_check_points/DCSCN/DCSCN_10epos.pt')
torch.save(optimizer.state_dict(), './model_check_points/DCSCN/DCSCN_optim_10epos.pt')
