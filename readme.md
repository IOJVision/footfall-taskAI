# Custom Tag Detection with YOLOv5

This repository showcases a project that applies a custom YOLOv5 model to detect and count people tags. The model counts only the detected tags that have confidence over 0.5.

## Solution Prediction Video

This is a gif preview showing the solution prediction of the detected tags:

![Solution Prediction Gif](comp_solution_pred.gif)

For the full video, please access it [here](https://github.com/IOJVision/footfall-taskAI/blob/main/comp_solution_pred.mp4).


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

# Custom Predict.py Usage

To leverage the customized predict.py script used in this repository for tag detection:

1. Clone the original YOLOv5 repository by Ultralytics. Run the following command in your terminal:

```bash
git clone https://github.com/ultralytics/yolov5
```

2. Download the custom [predict.py](https://github.com/IOJVision/footfall-taskAI/blob/main/predict.py) script from this repository.

3. Replace the original `predict.py` located in the `yolov5/segment` directory of the cloned YOLOv5 repository with the custom `predict.py` script you've just downloaded. 

Now, you are ready to use the YOLOv5 with the custom `predict.py` for improved tag detection performance.


```bash
python predict.py --weights /content/drive/MyDrive/footfall/yolov5/runs/train-seg/exp3/weights/best.pt --conf 0.25 --source /content/drive/MyDrive/footfall/sample.mp4 --name exp1805
```


Here's how you can write the instructions to modify the `data.yaml` file for users to fit their specific needs:


## Customizing [data.yaml](https://github.com/IOJVision/footfall-taskAI/blob/main/AI-Task-footfallcam-3/data.yaml) in AI-Taks-footfallcam-3

The provided `data.yaml` is configured for the specific use-case of this repository, but you may need to modify it to fit your specific needs. 

```yaml
names:
- work-tag
nc: 1
roboflow:
  license: CC BY 4.0
  project: ai-task-footfallcam
  url: https://universe.roboflow.com/azri-izzul-4qda7/ai-task-footfallcam/dataset/3
  version: 3
  workspace: azri-izzul-4qda7
test: /content/drive/MyDrive/footfall/AI-Task-footfallcam-3/test/images
train: /content/drive/MyDrive/footfall/AI-Task-footfallcam-3/train/images
val: /content/drive/MyDrive/footfall/AI-Task-footfallcam-3/valid
```

In the above `data.yaml` file:

- `names` should be a list of class names in your dataset.
- `nc` should be the number of classes in your dataset.
- The `roboflow` section contains metadata about the project and can be left as is.
- `test`, `train`, and `val` should point to the respective directories that contain your testing, training, and validation image sets. Update these to point to the correct paths if your directories are different.

Once you've edited your `data.yaml` file, save it and you're ready to proceed!

---

If you have any issues, please create a new issue in the repository. We welcome pull requests and any other contributions.

---

