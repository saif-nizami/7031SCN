CONFIG = {
    "model_def": {1: "YOLOv8", 2: "Faster R-CNN", 3: "DETR"},
    "model_name": "yolov8n.pt",
    "data_yaml": "data/coco_yolo_subset/dataset.yaml",
    "epochs": 5,             
    "img_size": 640,
    "batch_size": 8,         
    "device": "cuda",        
    "project": "outputs",
    "experiment_name": "yolov8_exp",
    "exist_ok": True,
    "seed": 42
}