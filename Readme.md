
# YOLO Object Detection Model Training

This repository contains code to train a YOLO (You Only Look Once) object detection model using a custom dataset. The training pipeline utilizes the Ultralytics YOLOv8 framework for fast and efficient object detection.

## 🧠 Model

This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) implementation, which supports various YOLO model sizes including `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, and `yolov8x`.

## 📁 Project Structure

```
.
├── YOLO_Train_model.ipynb  # Main notebook to train the model
├── dataset/
│   ├── images/
│   └── labels/
└── ...
```

## 🧰 Requirements

Install the necessary dependencies:

```bash
pip install -U ultralytics
```

Optional (for Google Colab users):

```bash
!pip install roboflow
```

## 📦 Dataset

The training process is designed for datasets structured in YOLO format. If you're using Roboflow or a similar platform, export the dataset in YOLOv5 PyTorch format and unzip it into your working directory.

Ensure the directory structure looks like:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## ⚙️ Training Configuration

To train a model, update the following configurations in the notebook:

- `data.yaml` path
- Choose the model size: `'yolov8n.pt'`, `'yolov8s.pt'`, `'yolov8m.pt'`, `'yolov8l.pt'`, `'yolov8x.pt'`
- Batch size, epochs, image size, etc.

Example training command used in the notebook:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="path/to/data.yaml", epochs=20, batch=16, imgsz=640)
```

## 📈 Results

After training, results are saved in the `runs/detect/` directory, including:

- Training plots
- Best model weights (`best.pt`)
- Training metrics (mAP, precision, recall)

## 🔍 Evaluation

Use the validation dataset to evaluate the model's performance:

```python
model.val()
```

## 🧪 Inference

You can run inference on images using:

```python
results = model("path/to/image.jpg")
results.show()
```

## 💾 Exporting the Model

Export the trained model to different formats:

```python
model.export(format='onnx')  # Options: 'onnx', 'torchscript', 'coreml', etc.
```

## 📌 Notes

- Ensure your dataset annotations follow YOLO format.
- Adjust the training hyperparameters based on your dataset size and complexity.

## 📄 License

This project is licensed under the MIT License.

---

Happy training! 🚀
