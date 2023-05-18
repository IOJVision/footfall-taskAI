# Custom Tag Detection with YOLOv5

This repository showcases a project that applies a custom YOLOv5 model to detect and count people tags. The model counts only the detected tags that have confidence over 0.5.

## Solution Prediction Video

This is the video showing the solution prediction of the detected tags.

<video width="960" height="720" controls>
  <source src="comp_solution_pred.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

## Setup and Installation

1. Navigate to your project directory.

```bash
cd /content/drive/MyDrive/footfall
```

2. Clone the YOLOv5 repository and install the required dependencies.

```bash
cd yolov5
pip install -r requirements.txt
```

## Training the Model

1. First, we need to import necessary modules.

```python
import torch
import utils
display = utils.notebook_init()
```

2. Then, navigate to the segmentation sub-directory and start training the model.

```bash
cd yolov5/segment
python train.py --img 640 --batch 128 --epochs 50 --data /content/drive/MyDrive/footfall/AI-Task-footfallcam-3/data.yaml --weights yolov5s-seg.pt --name exp3
```

3. Upon completion of the training, visualize the results.

```python
display.Image(filename='/content/drive/MyDrive/footfall/yolov5/runs/train-seg/exp3/results.png', width=1200)
```

## Validation and Prediction

1. Validate the trained model.

```bash
python val.py --weights /content/drive/MyDrive/footfall/yolov5/runs/train-seg/exp3/weights/best.pt --data /content/drive/MyDrive/footfall/AI-Task-footfallcam-2/data.yaml --img 640  --name exp3
```

2. Predict using the trained model.

```bash
python predict.py --weights /content/drive/MyDrive/footfall/yolov5/runs/train-seg/exp3/weights/best.pt --conf 0.25 --source /content/drive/MyDrive/footfall/sample.mp4 --name exp1805
```

---

If you have any issues, please create a new issue in the repository. We welcome pull requests and any other contributions.

---

