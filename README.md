# Waste YOLOv8n Model

Ultralytics YOLOv8n object detection model for waste detection.

## Files

- `models/best.pt`: trained YOLOv8n weights
- `models/model_info.json`: model name, classes, input size, and inference defaults
- `predict_image.py`: image prediction script with filled bounding boxes
- `training/train_yolo_waste.py`: training script used to create this model
- `requirements.txt`: Python dependencies

## Classes

| ID | Class |
|---:|---|
| 0 | Aluminum can |
| 1 | Cardboard |
| 2 | Container for household chemicals |
| 3 | Glass bottle |
| 4 | Organic |
| 5 | Paper |
| 6 | Plastic bag |
| 7 | Plastic bottle |
| 8 | Plastic cup |
| 9 | Tin |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Predict

```powershell
.\.venv\Scripts\python.exe predict_image.py --source "C:\path\to\image.jpg"
```

Folder input is also supported:

```powershell
.\.venv\Scripts\python.exe predict_image.py --source "C:\path\to\images"
```

Results are saved to:

```text
outputs\
```

Use a lower confidence threshold when the object is real-world phone footage and detections are weak:

```powershell
.\.venv\Scripts\python.exe predict_image.py --source "C:\path\to\image.jpg" --conf 0.10
```

## Python Usage

```python
import json
from pathlib import Path
from ultralytics import YOLO

repo_dir = Path(__file__).resolve().parent
info = json.loads((repo_dir / "models" / "model_info.json").read_text(encoding="utf-8"))

model = YOLO(repo_dir / info["model"]["weights_path"])
results = model.predict(
    source="image.jpg",
    imgsz=info["input"]["image_size"],
    conf=info["thresholds"]["default_confidence"],
)

class_names = info["classes"]["names"]
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(class_id, class_name, confidence, xyxy)
```

## Model Info

- Architecture: YOLOv8n
- Framework: Ultralytics YOLO
- Input size: `416`
- Default confidence: `0.25`
- Default IoU: `0.7`
- Output: bounding boxes with `class_id`, `class_name`, `confidence`, and `xyxy`

This repository is intended for inference and deployment. Training artifacts and epoch checkpoints are intentionally excluded.

## Recreate Training

The included training script expects the original Roboflow YOLOv8 dataset in YOLO format.

Example command used for this model family:

```powershell
python training\train_yolo_waste.py --model yolov8n.pt --epochs 150 --batch 16 --augmentation light --name waste_yolov8n_full_aug150
```

If the dataset is not in the original local path, pass `--data` explicitly:

```powershell
python training\train_yolo_waste.py --data "C:\path\to\data.yaml" --model yolov8n.pt --epochs 150 --batch 16 --augmentation light --name waste_yolov8n_full_aug150
```
