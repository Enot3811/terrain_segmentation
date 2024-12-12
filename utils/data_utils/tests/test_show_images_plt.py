"""Test script for show_images_plt function."""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.functions import show_images_plt


if __name__ == '__main__':
    np.random.seed(42)
    normal_image = np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8)
    one_ch_image = np.random.randint(0, 256, (448, 448, 1), dtype=np.uint8)
    flat_image = np.random.randint(0, 256, (448, 448), dtype=np.uint8)
    big_image = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
    show_images_plt(flat_image, plt_show=False, title='flat image')
    show_images_plt([big_image, normal_image], plt_show=False,
                    subtitles=['big image', 'normal image'])
    show_images_plt([normal_image, normal_image, normal_image, one_ch_image,
                     flat_image, big_image],
                    subtitles=['first', 'second', 'third', 'one channel',
                               'flat image', 'big image'],
                    title='all images',
                    figsize=(16, 16),
                    grid_size=(3, 2),
                    colormap='gray',
                    plt_show=True)
