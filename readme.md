# Object Detection using Neural Networks (Task A)

This project implements and compares three object detection models:

- YOLOv8 (Ultralytics)
- Faster R-CNN (Torchvision)
- DETR (Transformer-based)

The system uses a **menu-driven pipeline (main.py)** to train and evaluate each model.

---

## 🚀 Features

- Train and evaluate 3 models from a single entry point
- COCO-format dataset support
- COCO evaluation metrics (mAP, IoU, etc.)
- Modular structure for easy experimentation

---

## 📁 Project Structure

.
├── main.py
├── configs/
│   └── config.py
├── models/
│   ├── yolov8/
│   ├── faster_rcnn/
│   ├── detr/
├── utils/
├── data/
├── outputs/
├── runs/
└── requirements.txt

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/object-detection-task-a.git
cd object-detection-task-a

Create virtual environment:

python -m venv venv

Activate environment:

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## 📊 Dataset Setup

Expected COCO-style dataset structure:

data/
└── coco_faster_rcnn_subset/
    ├── val2017/
    └── annotations/
        └── instances_val2017.json

For YOLOv8:

data/coco_yolo/

Update paths inside:
configs/config.py

---

## ▶️ Usage

Run the main script:

python main.py

---

## 🧭 Menu Flow

Step 1: Select Model

1. YOLOv8
2. Faster R-CNN
3. DETR
4. Exit

Step 2: Select Operation

1. Train
2. Evaluate
3. Exit

---

## 🔍 Model Workflows

### YOLOv8

Train:
- Trains model using CONFIG
- Saves weights to:
  runs/detect/outputs/yolov8_exp/weights/best.pt

Evaluate:
- Loads trained weights
- Runs validation
- Saves metrics to:
  results/yolov8_metrics.json

Prediction:
- Runs inference on validation images

---

### Faster R-CNN

Train:
- Saves model to:
  outputs/faster_rcnn.pth

Evaluate:
- Uses COCO evaluation
- Requires trained weights

---

### DETR

Train:
- Saves model to:
  outputs/detr.pth

Evaluate:
- Loads pretrained DETR backbone
- Applies trained weights
- Uses COCO evaluation

---

## 📈 Evaluation Metrics

- Mean Average Precision (mAP)
- Intersection over Union (IoU)
- Precision & Recall

---

## ⚠️ Important Notes

- Train the model before running evaluation
- Ensure dataset paths are correct
- DETR uses:
  facebook/detr-resnet-50
- num_labels must match dataset (COCO = 91)

---

## 💾 Outputs

YOLOv8:
runs/detect/

Faster R-CNN:
outputs/faster_rcnn.pth

DETR:
outputs/detr.pth

Metrics:
results/

---

## 🧪 Example Run

python main.py

Then select:
1 → YOLOv8
1 → Train

---

## 🧰 Tech Stack

- PyTorch
- Torchvision
- Ultralytics YOLOv8
- Hugging Face Transformers
- pycocotools

---

## 👨‍💻 Author

Saif Nizami

---

## 📄 License

For academic use only