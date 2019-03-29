# Model Specifications 


```python
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
model_cran_v2 = model_cran_v2.float() # if use cpu 

````
