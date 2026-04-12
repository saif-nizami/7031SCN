# CONFIG = {
#     "model_name": "yolov8n.pt",
#     "data_yaml": "coco128.yaml", #"data/coco_yolo/dataset.yaml",
#     "epochs": 1,
#     "img_size": 640,
#     "batch_size": 16,
#     "device": 0,
#     "project": "outputs",
#     "experiment_name": "yolov8_exp",
#     "seed": 42
# }

CONFIG = {
    "model_name": "yolov8n.pt",
    "data_yaml": "data/coco_yolo_subset/dataset.yaml",
    "epochs": 1,              # 🔥 increase from 1 (1 is too low)
    "img_size": 640,
    "batch_size": 8,          # safer for MPS
    "device": "mps",          # ✅ FIXED
    "project": "outputs",
    "experiment_name": "yolov8_exp",
    "exist_ok": True,         # ✅ prevent folder duplication
    "seed": 42
}