# Repository description
This repository is dedicated to the development of terrain segmentation models.
As a source of data I use satellite images and OSM masks that can be downloaded
from QGIS.


# Directory structure

## qgis
Scripts that interact with QGIS API.

## utils
Package that contains helper tools.

## unet
Experiment with handwritten UNet.

## cloud_cls
Small classification sub-task to filter out clouds and blurry images by using
YOLOv8-cls model.
