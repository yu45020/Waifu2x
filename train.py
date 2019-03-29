from apex.fp16_utils import FP16_Optimizer
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import trange

from Dataloader import *
from utils import image_quality
from utils.cls import CyclicLR
from utils.prepare_images import *

train_folder = './dataset/train'
test_folder = "./dataset/test"

img_dataset = ImageDBData(db_file='dataset/images.db', db_table="train_images_size_128_noise_1_rgb", max_images=24)
img_data = DataLoader(img_dataset, batch_size=6, shuffle=True, num_workers=6)

total_batch = len(img_data)
print(len(img_dataset))

test_dataset = ImageDBData(db_file='dataset/test2.db', db_table="test_images_size_128_noise_1_rgb", max_images=None)
num_test = len(test_dataset)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

criteria = nn.L1Loss()

model = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                single_conv_size=3, single_conv_group=1,
                scale=2, activation=nn.LeakyReLU(0.1),
                SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

model.total_parameters()


# model.initialize_weights_xavier_uniform()

# fp16 training is available in GPU only
model = network_to_half(model)
model = model.cuda()
model.load_state_dict(torch.load("CARN_model_checkpoint.pt"))

learning_rate = 1e-4
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
# optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True, weight_decay=weight_decay, lr=learning_rate)

optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0, verbose=False)
# optimizer.load_state_dict(torch.load("CARN_adam_checkpoint.pt"))

last_iter = -1  # torch.load("CARN_scheduler_last_iter")
scheduler = CyclicLR(optimizer.optimizer, base_lr=1e-4, max_lr=1e-4,
                     step_size=3 * total_batch, mode="triangular",
                     last_batch_iteration=last_iter)
train_loss = []
train_ssim = []
train_psnr = []

test_loss = []
test_ssim = []
test_psnr = []

# train_loss = torch.load("train_loss.pt")
# train_ssim = torch.load("train_ssim.pt")
# train_psnr = torch.load("train_psnr.pt")
#
# test_loss = torch.load("test_loss.pt")
# test_ssim = torch.load("test_ssim.pt")
# test_psnr = torch.load("test_psnr.pt")


counter = 0
iteration = 2
ibar = trange(iteration, ascii=True, maxinterval=1, postfix={"avg_loss": 0, "train_ssim": 0, "test_ssim": 0})
for i in ibar:
    # batch_loss = []
    # insample_ssim = []
    # insample_psnr = []
    for index, batch in enumerate(img_data):
        scheduler.batch_step()
        lr_img, hr_img = batch
        lr_img = lr_img.cuda().half()
        hr_img = hr_img.cuda()

        # model.zero_grad()
        optimizer.zero_grad()
        outputs = model.forward(lr_img)
        outputs = outputs.float()
        loss = criteria(outputs, hr_img)
        # loss.backward()
        optimizer.backward(loss)
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        counter += 1
        # train_loss.append(loss.item())

        ssim = image_quality.msssim(outputs, hr_img).item()
        psnr = image_quality.psnr(outputs, hr_img).item()

        ibar.set_postfix(ratio=index / total_batch, loss=loss.item(),
                         ssim=ssim, batch=index,
                         psnr=psnr,
                         lr=scheduler.current_lr
                         )
        train_loss.append(loss.item())
        train_ssim.append(ssim)
        train_psnr.append(psnr)

        # +++++++++++++++++++++++++++++++++++++
        #      save checkpoints by iterations
        # -------------------------------------

        if (counter + 1) % 500 == 0:
            torch.save(model.state_dict(), 'CARN_model_checkpoint.pt')
            torch.save(optimizer.state_dict(), 'CARN_adam_checkpoint.pt')
            torch.save(train_loss, 'train_loss.pt')
            torch.save(train_ssim, "train_ssim.pt")
            torch.save(train_psnr, 'train_psnr.pt')
            torch.save(scheduler.last_batch_iteration, "CARN_scheduler_last_iter.pt")

    # +++++++++++++++++++++++++++++++++++++
    #           End of One Epoch      
    # -------------------------------------

    # one_ite_loss = np.mean(batch_loss)
    # one_ite_ssim = np.mean(insample_ssim)
    # one_ite_psnr = np.mean(insample_psnr)

    # print(f"One iteration loss {one_ite_loss}, ssim {one_ite_ssim}, psnr {one_ite_psnr}")
    # train_loss.append(one_ite_loss)
    # train_ssim.append(one_ite_ssim)
    # train_psnr.append(one_ite_psnr)

    torch.save(model.state_dict(), 'CARN_model_checkpoint.pt')
    # torch.save(scheduler, "CARN_scheduler_optim.pt")
    torch.save(optimizer.state_dict(), 'CARN_adam_checkpoint.pt')
    torch.save(train_loss, 'train_loss.pt')
    torch.save(train_ssim, "train_ssim.pt")
    torch.save(train_psnr, 'train_psnr.pt')
    # torch.save(scheduler.last_batch_iteration, "CARN_scheduler_last_iter.pt")

    # +++++++++++++++++++++++++++++++++++++
    #           Test
    # -------------------------------------

    with torch.no_grad():
        ssim = []
        batch_loss = []
        psnr = []
        for index, test_batch in enumerate(test_data):
            lr_img, hr_img = test_batch
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()

            lr_img_up = model(lr_img)
            lr_img_up = lr_img_up.float()
            loss = criteria(lr_img_up, hr_img)

            save_image([lr_img_up[0], hr_img[0]], f"check_test_imgs/{index}.png")
            batch_loss.append(loss.item())
            ssim.append(image_quality.msssim(lr_img_up, hr_img).item())
            psnr.append(image_quality.psnr(lr_img_up, hr_img).item())

        test_ssim.append(np.mean(ssim))
        test_loss.append(np.mean(batch_loss))
        test_psnr.append(np.mean(psnr))

        torch.save(test_loss, 'test_loss.pt')
        torch.save(test_ssim, "test_ssim.pt")
        torch.save(test_psnr, "test_psnr.pt")

# import subprocess

# subprocess.call(["shutdown", "/s"])
