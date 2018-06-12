import matplotlib.pyplot as plt
import torch

loss_espcn_7 = torch.load("./model_results/ESPCN_7_6epos_loss.pk")
loss_espcn_7_nous = torch.load("./model_results/ESPCN_7_6epos_loss_noSU.pk")

plt.plot(loss_espcn_7, color='red', label='With Selection Unit')
plt.plot(loss_espcn_7_nous, color='green', label='No Selection Unit')
plt.title('ESPCN_7 Model 6 Epoches')
plt.xlabel('Per 50 images (96x96 -> 192x192)')
plt.ylabel('Weighted MSE Loss')
plt.legend()
plt.show()
