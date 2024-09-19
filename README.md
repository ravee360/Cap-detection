# Cap Detection using YOLOv8

This repository contains the code and model for detecting whether a person is wearing a cap or not using a YOLOv8-based object detection approach. The model was trained on a dataset with 671 cap-wearing images and 529 non-cap-wearing images, for a total of 1200 images, and trained over 200 epochs.

## Colab notebook link

[Colab notebook link](https://colab.research.google.com/drive/1z4-jNtFHVnYyat2ZM-qiNXq5PE2livBU?authuser=1#scrollTo=D0yu4cn_ruk1)

## Dataset Overview
he dataset consists of:

- **671 images of people wearing caps.
- **529 images of people without caps.
The dataset was split into training and validation sets for model training and evaluation. Annotations for the images were converted into YOLO format before training.

## Model Training
The model was trained using the YOLOv8 architecture for 200 epochs. The training was done on the normalized dataset, and the following metrics were achieved:

- Model: YOLOv8 (pre-trained on COCO dataset, fine-tuned for cap detection).
- Epochs: 200
- Optimizer: SGD (based on the chosen configuration).
- Batch size: 16.
- Input Image Size: 320x320
- Training Script: A custom training script was used to fine-tune the YOLOv8 model on the cap detection dataset.

##  Model Performance
After training the model for 200 epochs, the following metrics were achieved:

- Precision: 0.954
- Recall: 0.886
- mAP_50: 0.937 (Mean Average Precision at 50% IoU)
- mAP_50-95: 0.825 (Mean Average Precision across multiple IoU thresholds)
- Fitness: 0.836

 ## Speed:
- Preprocessing Time: 0.0681 seconds
- Inference Time: 0.9437 seconds
- Loss Time: 0.00047 seconds
- Postprocessing Time: 1.6046 seconds
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



