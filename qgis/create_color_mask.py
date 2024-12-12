from pathlib import Path

import numpy as np

from utils.data_utils.functions import read_image, save_image


orig_mask_dir = Path('~/exported_regions/osm/')
dest_mask_dir = Path('~/exported_regions/color_osm/')

colors = {
    0: np.array([255, 255, 255]),
    1: np.array([255, 0, 0]),
    2: np.array([0, 255, 0]),
    3: np.array([0, 0, 255])
}

for img_pth in orig_mask_dir.iterdir():
    img = read_image(img_pth, bgr_to_rgb=False)
    print(img.shape, np.unique(img))

    mask = img[..., 0]
    for key in colors:
        mask[mask == key]
        img[np.tile(mask, (1, 2))] = colors[key]
    
    save_image(img, dest_mask_dir / img_pth.with_suffix('.jpg').name)
