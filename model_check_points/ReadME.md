# Resume & Use Model Check Points 

This folder contains check points for models and their weights.  They are generated from [PyTorch's pickle](https://pytorch.org/docs/master/notes/serialization.html). 

Model specifications are in each folder's ReadME.  

Pickle names with "model" contain the entire models, and they can be used as an freeze module by calling the "forward_checkpoint" function to generate images. 

Example:
```python
import torch
# No need to reconstruct the model 
model = torch.load("./DCSCN/DCSCN_model_387epos_L12_noise_1.pt")
x = torch.randn((1,3,10,10)), torch.randn((1,3,20,20))
out = model.forward_checkpoint(a)
```

Pickle names with "weights" are model weights, and they are named dictionaries. 

Example:
```python
model = DCSCN(*)    # the setting must be the same to load check points weights. 
model.load_state_dict(torch.load("./DCSCN/DCSCN_weights_387epos_L12_noise_1.pt"))
# then you can resume the model training 
```

Model check poins in Upconv_7 and vgg_7 are from [waifu2x's repo](https://github.com/nagadomi/waifu2x/tree/master/models). To load weights into a model, please use ```load_pre_train_weights``` function. 

Example: 
```python
model = UpConv_7()
model.load_pre_train_weights(json_file=...)
# then the model is ready to use 
```
