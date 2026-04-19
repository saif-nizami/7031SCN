import sys

from configs.config import CONFIG

# YOLOv8
from models.yolov8.train import train_model as train_yolo
from models.yolov8.validate import validate_model as val_yolo
from models.yolov8.predict import run_inference as pred_yolo
from utils.save_metrics import save_metrics

# fasterRCNN
from models.faster_rcnn.train import train_model as train_frcnn

# DETR
from models.detr.train import train_model as train_detr

def run_yolov8():
    print("\nRunning YOLOv8 Pipeline...\n")

    # train_yolo(CONFIG)
    results = train_yolo(CONFIG)
    model_path = f"{results.save_dir}/weights/best.pt"
    print("Using model from:", model_path)
    # # model_path = f"{CONFIG['project']}/{CONFIG['experiment_name']}/weights/best.pt"

    metrics = val_yolo(model_path)
    save_metrics(metrics, "yolov8")
    pred_yolo(model_path, "data/coco_yolo/images/val")

def run_fasterrcnn():
    print("\nRunning Faster R-CNN Pipeline...\n")
    train_frcnn(CONFIG)

def run_detr():
    print("\nRunning DETR Pipeline...\n")
    train_detr(CONFIG)

def show_menu():
    print("\n==============================")
    print(" Select Model to Run ")
    print("==============================")
    print("1. YOLOv8")
    print("2. Faster R-CNN")
    print("3. DETR")
    print("4. Exit")
    print("==============================")

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            CONFIG["experiment_name"] = "yolov8_exp"
            run_yolov8()

        elif choice == "2":
            # print("\nFaster R-CNN not implemented yet\n")
            # sys.exit(130)
            CONFIG["experiment_name"] = "faster_rcnn"
            run_fasterrcnn()

        elif choice == "3":
            # print("\nDETR not implemented yet\n")
            # sys.exit(130)
            CONFIG["experiment_name"] = "detr"
            run_detr()

        elif choice == "4":
            print("\n👋 Exiting...")
            sys.exit(0)

        else:
            print("\nInvalid choice, try again!\n")


if __name__ == "__main__":
    main()