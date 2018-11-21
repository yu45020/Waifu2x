from torchvision.utils import save_image

from utils.prepare_images import *

lr = "./benchmark/miku_small.png"
pre_trained = "model_results/DCSCN_model_387epos.pt"
model = DCSCN(color_channel=3,
              up_scale=2,
              feature_layers=12,
              first_feature_filters=196,
              last_feature_filters=48,
              reconstruction_filters=128,
              up_sampler_filters=32)
model.load_state_dict(torch.load(pre_trained, map_location='cpu'))

img = Image.open(lr).convert("RGB")
img_up = img.resize((2 * img.size[0], 2 * img.size[1]), Image.BILINEAR)
img = to_tensor(img).unsqueeze(0)
img_up = to_tensor(img_up).unsqueeze(0)

if torch.cuda.is_available():
    model = model.cuda()
    img = img.cuda()
    img_up = img_up.cuda()

out = model((img, img_up))
save_image(out, './benchmark/miku_dcscn.png')
