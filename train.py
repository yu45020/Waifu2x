import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import trange

from Dataloader import *
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

# img_dataset = ImageData(train_folder,
#                         patch_size=128,
#                         shrink_size=2,
#                         noise_level=1,
#                         down_sample_method=None,
#                         color_mod='RGB', dummy_len=None)
img_dataset = ImageH5Data('train_images.hdf5', 'shrink_2_noise_level_1_downsample_random_rgb')
img_data = DataLoader(img_dataset, batch_size=3, shuffle=True)

total_batch = len(img_data)
len(img_dataset)

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

# model = DCSCN(color_channel=3,
#               up_scale=2,
#               feature_layers=12,
#               first_feature_filters=196,
#               last_feature_filters=48,
#               reconstruction_filters=128,
#               up_sampler_filters=32)
model = CARNV2(color_channels=3, scale=2, activation=nn.LeakyReLU(0.1), SEBlock=True, conv=nn.Conv2d,
               atrous=(1, 2, 4, 2, 1))
# model = CARN(color_channels=3, scale=2, activation=nn.LeakyReLU(0.1), num_blocks=5)
model.total_parameters()
# model.load_state_dict(torch.load("model_param_full_v2_partial.pkl"))
# pre_train = torch.load("./model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt", map_location='cpu')
# new = model.state_dict()
# a = {}
# for i, j in zip(pre_train, new):
#     a[j] = pre_train[i]
#
# model.load_state_dict(pre_train, strict=False)

# torch.save(model.state_dict(), "./model_results/DCSCN_model_135epos_2.pt")

learning_rate = 5e-4
weight_decay = 1e-5
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
model = model.cuda()
# criteria = criteria.cuda()
optimizer.param_groups[0]['lr'] = 1e-4

iteration = 500
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
        # lr_up = nn.functional.interpolate(lr_img, scale_factor=2,
        #                                   mode='bicubic', align_corners=False)
        model.zero_grad()
        outputs = model.forward(lr_img)
        loss = criteria(outputs, hr_img)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        counter += 1
        # all_loss.append(loss.item())
        batch_loss.append(loss.item())
        ssim = image_quality.msssim(outputs, hr_img)
        # ibar.set_postfix(ratio=index / total_batch, loss=loss.item(), ssim=ssim, batch=index)
        insample_ssim += ssim

    one_ite_loss = np.mean(batch_loss)
    print("One iteration loss {:.5f}".format(one_ite_loss))
    avg_ssim.append(insample_ssim)
    all_loss.append(one_ite_loss)
    # eval
    # ssim = 0
    # for test_batch in test_data:
    #     lr_img, hr_img = test_batch
    #     lr_img_up = model(lr_img.cuda())
    #     ssim += image_quality.msssim(lr_img_up, hr_img.cuda())
    # avg_ssim.append(ssim / num_test)
    # ibar.write("Average loss {:.5f}; train SSIM {:.5f}; test SSIM {:.5f}".format(one_ite_loss,
    #                                                                              insample_ssim / total_batch,
    #                                                                              ssim / len(test_data)))

# torch.save(model.state_dict(), './model_check_points/DCSCN/DCSCN_10epos.pt')
# torch.save(optimizer.state_dict(), './model_check_points/DCSCN/DCSCN_optim_10epos.pt')
import matplotlib.pylab as plt

plt.plot(all_loss[10:])
# plt.plot([all_loss[i] for i in range(100, 10000, 100)])
plt.show()

save_image(torch.cat([outputs, hr_img], dim=0), 'up.png')

torch.save(all_loss, 'loss_full_v2_se_atros_12421.pkl')  # loss_full_v2_se_partial_atros_12421
torch.save(model.state_dict(), 'model_full_v2_se_atros_12421.pkl')  # model_full_v2_se_partial_atros_121

loss_standard = torch.load("loss_standard.pkl")
loss_standard_5 = torch.load("loss_standard_5.pkl")

loss_se = torch.load('loss_full_v2_se.pkl')

loss_se_partial_atrous_121 = torch.load("loss_full_v2_se_partial_atros_121.pkl")
loss_se_partial_atrous_421 = torch.load("loss_full_v2_se_partial_atros_421.pkl")
loss_se_partial_atrous_12421 = torch.load("loss_full_v2_se_partial_atros_12421.pkl")

loss_se_atrous_121 = torch.load("loss_full_v2_se_atros_121.pkl")

plt.plot(loss_standard, label='Standard', linestyle='-')
plt.plot(loss_standard_5, label='Standard 5 Blocks', linestyle='-')

plt.plot(loss_se, label='Standard SE', linestyle='-.')

plt.plot(loss_se_partial_atrous_121, label='SE Partial Atrous 1-2-1', linestyle='-.')
plt.plot(loss_se_partial_atrous_421, label='SE Partial Atrous 4-2-1', linestyle='--')
plt.plot(loss_se_partial_atrous_12421, label='SE Partial Atrous 1-2-4-2-1')
plt.plot(loss_se_atrous_121, label='SE  Atrous 1-2-1')

plt.ylim(bottom=0, top=0.03)
plt.legend(loc='upper right')
plt.grid()
plt.show()
plt.close()

# plt.close()
