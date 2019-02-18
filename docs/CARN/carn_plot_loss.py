import torch
import matplotlib.pyplot as plt
import pandas as pd

train_loss_128 = torch.load("check_points/input_size_weight_decay/128_weight_decay_1e-5/train_loss.pt")
test_loss_128 = torch.load("check_points/input_size_weight_decay/128_weight_decay_1e-5/test_loss.pt")

train_loss_64 = torch.load("check_points/input_size_weight_decay/64_weight_decay_1e-5/train_loss.pt")
test_loss_64 = torch.load("check_points/input_size_weight_decay/64_weight_decay_1e-5/test_loss.pt")

plt.plot(train_loss_128, label="128x128 input size train loss")
plt.plot(train_loss_64, label="64x64 input size train loss")
plt.plot(test_loss_128, label="128x128 input size test loss")
plt.plot(test_loss_64, label="64x64 input size test loss")

plt.grid()
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")

plt.savefig("docs/CARN/plots/128_vs_64_model_loss.png")
plt.show()

train_ssim_128 = torch.load("check_points/input_size_weight_decay/128_weight_decay_1e-5/train_ssim.pt")
train_ssim_64 = torch.load("check_points/input_size_weight_decay/64_weight_decay_1e-5/train_ssim.pt")
test_ssim_128 = torch.load("check_points/input_size_weight_decay/128_weight_decay_1e-5/test_ssim.pt")
test_ssim_64 = torch.load("check_points/input_size_weight_decay/64_weight_decay_1e-5/test_ssim.pt")

plt.plot(train_ssim_128, label="128x128 input size train ssim")
plt.plot(train_ssim_64, label="64x64 input size train ssim")
plt.plot(test_ssim_128, label="128x128 input size test ssim")
plt.plot(test_ssim_64, label="64x64 input size test ssim")

plt.grid()
plt.legend()
plt.title("SSIM")
plt.xlabel("Epoch")
plt.savefig("docs/CARN/plots/128_vs_64_model_ssim.png")

plt.show()
