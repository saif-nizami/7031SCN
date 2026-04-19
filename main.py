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

def show_main_menu():
    print("\n==============================")
    print(" Select Model")
    print("==============================")
    print("1. YOLOv8")
    print("2. Faster R-CNN")
    print("3. DETR")
    print("4. Exit")
    print("==============================")

def show_L2_menu():
    print("\n==============================")
    print(" Select Operation")
    print("==============================")
    print("1. Train")
    print("2. Evaluate")
    print("3. Exit")
    print("==============================")

def main_runner(first_choice, second_choice):
    if first_choice == "1":
        print("YOLOv8 Pipeline Triggered!")
        if second_choice == "1":
            print("YOLOv8 Training!")
        elif second_choice == "2":
            print("YOLOv8 Evaluating!")
        else:
            sys.exit(0)
    
    elif first_choice == "2":
        print("Faster R-CNN Pipeline Triggered!")
        if second_choice == "1":
            print("Faster R-CNN Training!")
        elif second_choice == "2":
            print("Faster R-CNN Evaluating!")
        else:
            sys.exit(0)
    
    elif first_choice == "3":
        print("DETR")
        if second_choice == "1":
            print("DETR Training!")
        elif second_choice == "2":
            print("DETR Evaluating!")
        else:
            sys.exit(0)
    
    else:
        print("Exit")
        sys.exit(0)


def main():
    # while True:
        show_main_menu()
        first_choice = input("Enter your model choice (1-4): ").strip()

        if first_choice == "1":
            CONFIG["experiment_name"] = "yolov8_exp"
            show_L2_menu()
            second_choice = input("Enter your model operation (1-3): ").strip()
            main_runner(first_choice, second_choice)
            # run_yolov8()

        elif first_choice == "2":
            CONFIG["experiment_name"] = "faster_rcnn"
            show_L2_menu()
            second_choice = input("Enter your model operation (1-3): ").strip()
            main_runner(first_choice, second_choice)
            # run_fasterrcnn()

        elif first_choice == "3":
            CONFIG["experiment_name"] = "detr"
            show_L2_menu()
            second_choice = input("Enter your model operation (1-3): ").strip()
            main_runner(first_choice, second_choice)
            # run_detr()

        elif first_choice == "4":
            print("\n👋 Exiting...")
            sys.exit(0)

        else:
            print("\nInvalid choice, try again!\n")


if __name__ == "__main__":
    main()