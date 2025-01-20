import segmentation_models_pytorch as smp
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from utils.torch_utils.functions import convert_seg_mask_to_one_hot


n_classes = 5
b_size = 2

loss_fn = smp.losses.DiceLoss('multiclass')

logits = torch.randn(b_size, n_classes, 224, 224)
targets = torch.randint(0, n_classes, (b_size, 224, 224))
loss = loss_fn(logits, targets)
print(loss)
