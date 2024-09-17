# Cap Detection using YOLOv8

This repository contains a machine learning project focused on detecting whether a person is wearing a cap using a model trained on the YOLOv8 architecture. The project includes the trained model in `.pt` format, a Python script (`python_cap_detection.py`) for performing real-time detection using OpenCV, and the necessary dependencies listed in `requirements.txt`.

## Dataset Overview

The model is trained on a dataset from the [Cap Dataset on Kaggle](https://www.kaggle.com/datasets/shivanandverma/cap-dataset), which consists of 671 annotated images containing people wearing and not wearing caps. The images were pre-annotated in XML format, which was then converted to YOLO format for training.

### Annotation Conversion Process

The XML annotations provided in the dataset were structured as follows:

```python
labels_dict = dict(
    img_path=[],
    xmin=[],
    xmax=[],
    ymin=[],
    ymax=[],
    img_w=[],
    img_h=[]
)
