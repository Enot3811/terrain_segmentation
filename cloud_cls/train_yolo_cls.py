from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(
    data='/home/pc0/projects/data/qgis/blur_clouds_cls/',
    epochs=10,
    project='region_localizer/cloud_cls/trains',
    name='blur_clouds_cls',
    batch=32)
