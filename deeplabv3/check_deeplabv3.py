"""Load and check torch DeeplabV3 implementation."""


import sys
from pathlib import Path

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parents[1]))
from utils.data_utils.functions import (
    read_image, generate_class_to_colors, show_images_cv2)
from utils.torch_utils.functions import convert_seg_mask_to_color


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'deeplabv3_resnet50',
        weights='DEFAULT',
    )
    model.eval()
    model.to(device)

    print('Model architecture:')
    print(model)
    
    # Get number of classes and generate colors for mask
    n_classes = model.classifier[4].out_channels
    class_to_colors = generate_class_to_colors(n_classes)

    # Read image and preprocess image
    image = read_image('data/deeplab1.png')
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transform_image = transform(image=image)['image']
    transform_image = transform_image.unsqueeze(0)

    print('Image shape:')
    print(transform_image.shape)

    # Forward pass
    with torch.no_grad():
        output = model(transform_image.to(device))
    output = output['out']
    print('Output shape:')
    print(output.shape)

    # Postprocess output
    out_seg_mask = torch.argmax(output, dim=1).cpu().numpy().squeeze()
    color_mask = convert_seg_mask_to_color(out_seg_mask, class_to_colors)
    resize_if_needed = A.LongestMaxSize(max_size=1000)
    image = resize_if_needed(image=image)['image']
    color_mask = resize_if_needed(image=color_mask)['image']
    
    # Show images
    show_images_cv2([image, color_mask],
                    window_title=['Image', 'Segmentation mask'])
    
    # Replace last layer to change number of classes
    new_classes = 4
    in_features = model.classifier[4].in_channels
    kernel_size = model.classifier[4].kernel_size
    stride = model.classifier[4].stride
    padding = model.classifier[4].padding
    dilation = model.classifier[4].dilation
    padding_mode = model.classifier[4].padding_mode
    bias = not model.classifier[4].bias is None

    model.classifier[4] = nn.Conv2d(
        in_features, new_classes, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, padding_mode=padding_mode,
        bias=bias
    ).to(device)
    model.aux_classifier[4] = nn.Conv2d(
        in_features, new_classes, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, padding_mode=padding_mode,
        bias=bias
    ).to(device)
    
    # Test new model
    with torch.no_grad():
        output_dict = model(transform_image.to(device))
    output = output_dict['out']
    aux_output = output_dict['aux']
    print('New output shape in eval mode:')
    print(output.shape, aux_output.shape)

    # Test new model in train mode
    model.train()
    batch = torch.cat((transform_image, transform_image), dim=0).to(device)
    output_dict = model(batch)
    output = output_dict['out']
    aux_output = output_dict['aux']
    print('New output shape in train mode:')
    print(output.shape, aux_output.shape)


if __name__ == "__main__":
    main()
