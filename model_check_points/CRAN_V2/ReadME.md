# Model Specifications 


```python
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
model_cran_v2 = model_cran_v2.float() # if use cpu 

````

To use pre-trained model for training 

```python

model = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                single_conv_size=3, single_conv_group=1,
                scale=2, activation=nn.LeakyReLU(0.1),
                SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

model = network_to_half(model)
model = model.cuda()
model.load_state_dict(torch.load("CARN_model_checkpoint.pt"))

learning_rate = 1e-4
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0, verbose=False)
optimizer.load_state_dict(torch.load("CARN_adam_checkpoint.pt"))

last_iter = torch.load("CARN_scheduler_last_iter") # -1 if start from new
scheduler = CyclicLR(optimizer.optimizer, base_lr=1e-4, max_lr=4e-4,
                     step_size=3 * total_batch, mode="triangular",
                     last_batch_iteration=last_iter)
                     
```