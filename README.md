# Cap Detection using YOLOv8

This repository contains a machine learning project focused on detecting whether a person is wearing a cap using a model trained on the YOLOv8 architecture. The project includes the trained model in `.pt` format, a Python script (`python_cap_detection.py`) for performing real-time detection using OpenCV, and the necessary dependencies listed in `requirements.txt`.

## Colab notebook link

[Colab notebook link]([https://www.kaggle.com/code/ravee360/cap-detection/edit](https://colab.research.google.com/drive/1z4-jNtFHVnYyat2ZM-qiNXq5PE2livBU?authuser=1#scrollTo=D0yu4cn_ruk1)

## Dataset Overview

The model is trained on a dataset from the [Cap Dataset on Kaggle](https://www.kaggle.com/datasets/shivanandverma/cap-dataset), which consists of 671 annotated images containing people wearing caps. The images were pre-annotated in XML format, which was then converted to YOLO format for training.

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
# This structure was converted into YOLO format using the following formula for bounding box normalization:

x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
width = (row['xmax'] - row['xmin']) / row['img_w']
height = (row['ymax'] - row['ymin']) / row['img_h']
```
This conversion ensures that the bounding box coordinates are normalized with respect to the image dimensions, which is required by the YOLO format.
## Model Training
The model was trained using the YOLOv8 architecture for 200 epochs. The training was done on the normalized dataset, and the following metrics were achieved:

- **mAP_50** (Mean Average Precision at 50% IoU threshold): 0.82
- **mAP_50-95** (Mean Average Precision across different IoU thresholds): 0.651

The model was trained with the following key hyperparameters:

- **Batch Size**: 16
- **Epochs**: 200
  ## Project Structure

The repository contains the following files:

- **`best_model.pt`**: The trained YOLOv8 model file.
- **`python_cap_detection.py`**: Python script that uses OpenCV to load the trained model and perform real-time cap detection via webcam or video input.
- **`requirements.txt`**: A list of Python packages required to run the project.

### Python Script (`python_cap_detection.py`)

This script uses OpenCV to capture frames from a webcam and runs the cap detection model in real-time. Here's how the script works:

1. **Loads the YOLOv8 model** from the `.pt` file.
2. **Captures input** either from a live webcam feed.
3. **Performs detection** on each frame, drawing bounding boxes around detected caps.
4. **Displays the real-time feed** with the detection results.

After training, the best-performing model was saved as `best_model.pt`.

## Installation

### Step 1: Clone the Repository

To begin, clone the repository to your local machine:

```bash
git clone https://github.com/ravee360/Cap-detection.git
cd cap-detection
```
### Step 2: Install Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Model Inference Process

The script performs the following steps for model inference:

1. **Capture Frames**: Uses the `cv2.VideoCapture` function to capture frames from the webcam.
2. **Detection**: Each frame is passed through the YOLOv8 model to detect caps.
3. **Output**: If a cap is detected, the model outputs bounding boxes and confidence scores.
4. **Display**: The bounding boxes are drawn around detected caps, and the result is displayed on the screen in real-time.
5. **Exit**: To exit the real-time detection, press `q`.



