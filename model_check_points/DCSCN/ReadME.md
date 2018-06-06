# Model Specifications 

## 12 Layers Model

```python
model = DCSCN(color_channel=3,
              up_scale=2,
              feature_layers=12,
              first_feature_filters=196,
              last_feature_filters=48,
              reconstruction_filters=64,
              up_sampler_filters=32)
````
