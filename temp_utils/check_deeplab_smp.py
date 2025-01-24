import segmentation_models_pytorch as smp
import torch


preprocessing_params = smp.encoders.get_preprocessing_params(
    encoder_name='tu-xception71',
    pretrained='imagenet'
)
print(preprocessing_params)

model = smp.DeepLabV3Plus(
    encoder_name='tu-xception71',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
)

# print(model)


test_tensor = torch.randn(2, 3, 224, 224)

model.eval()
out = model(test_tensor)
print(out.shape)


model.train()
out = model(test_tensor)
print(out.shape)
