"""Check loading pipeline with given config."""


import sys
from pathlib import Path
from torch.utils.data import DataLoader
from cv2 import destroyAllWindows
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from utils.torch_utils.datasets import SegmentationDataset
from utils.data_utils.functions import show_images_cv2
from utils.train_utils import get_transforms, read_config


def main(config_path: str):
    
    # Read config
    config = read_config(config_path)

    # Get transforms
    if config['train_transforms']:
        train_transforms = get_transforms(config['train_transforms'])
    else:
        train_transforms = None
    if config['val_transforms']:
        val_transforms = get_transforms(config['val_transforms'])
    else:
        val_transforms = None

    # Mean and std for denormalization
    mean = np.array(config['train_transforms']['normalize']['params']['mean'])
    std = np.array(config['train_transforms']['normalize']['params']['std'])
    
    # Get datasets and loaders
    train_dset = SegmentationDataset(
        **config['train_dataset_params'], transforms=train_transforms)
    val_dset = SegmentationDataset(
        **config['val_dataset_params'], transforms=val_transforms)

    train_loader = DataLoader(train_dset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle_train'],
                              num_workers=config['num_workers'],
                              collate_fn=SegmentationDataset.collate_func)
    val_loader = DataLoader(val_dset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle_val'],
                            num_workers=config['num_workers'],
                            collate_fn=SegmentationDataset.collate_func)

    # Check loading pipeline
    for loader in (train_loader, val_loader):
        go_next_loader = False
        for batch in loader:
            images, masks, image_paths, mask_paths, shapes = batch

            for i in range(images.shape[0]):
                # Convert tensor to numpy array
                image = images[i].permute(1, 2, 0).numpy()
                mask = masks[i].numpy()

                print('Image min:', image.min(), ', Image max:', image.max())

                # Denormalize image
                image = ((image * std + mean) * 255).astype(np.uint8)
                images_to_show = [image]
                titles = ['image']

                # Show each mask separately
                if config['train_dataset_params']['one_hot_encoding']:
                    for j in range(mask.shape[0]):
                        mask_j = mask[j].astype(np.uint8)
                        mask_j = 255 * mask_j
                        images_to_show.append(mask_j)
                        titles.append(f'mask {j}')
                # Convert mask to color
                else:
                    color_mask = SegmentationDataset.seg_mask_to_color(
                        mask, train_dset.id_to_color)
                    images_to_show.append(color_mask)
                    titles.append('mask')

                # Show
                print(image_paths[i].name, mask_paths[i].name)
                key = show_images_cv2(
                    images_to_show, titles, destroy_windows=False)
                if key == 27:  # esc
                    destroyAllWindows()
                    return
                elif key == 13:  # enter
                    destroyAllWindows()
                    go_next_loader = True
                    break
            if go_next_loader:
                break
    destroyAllWindows()


if __name__ == '__main__':
    config_path = 'train_configs/test_config.yaml'
    main(config_path)
