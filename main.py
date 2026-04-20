import sys
import os
import torch
from models.faster_rcnn.dataset import COCODataset
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor

from configs.config import CONFIG

# YOLOv8
from models.yolov8.train import train_model as train_yolo
from models.yolov8.validate import validate_model as val_yolo
from models.yolov8.predict import run_inference as pred_yolo
from utils.save_metrics import save_metrics

# fasterRCNN
from models.faster_rcnn.train import train_model as train_frcnn
from models.faster_rcnn.main_eval import evaluate_model

# DETR
from models.detr.train import train_model as train_detr

def frcnn_collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return tuple(zip(*batch))

def frcnn_eval_helper():
    import torchvision.transforms as T

    def get_transform():
        return T.Compose([
        T.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/val2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_val2017.json",
        transforms=get_transform()
    )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=frcnn_collate_fn,
        num_workers=2
    )

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_val2017.json")

    # rebuild model
    num_classes = len(dataset.cat_ids)

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(torch.load("outputs/faster_rcnn.pth"))
    model.to(device)

    # mapping back to COCO category IDs
    label2cat = {v: k for k, v in dataset.cat2label.items()}

    metrics = evaluate_model(model, val_loader, coco_gt, device, label2cat)

class detr_CollateFn:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return [], []

        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        max_h = max(img.shape[1] for img in pixel_values)
        max_w = max(img.shape[2] for img in pixel_values)

        padded_images = []
        for img in pixel_values:
            c, h, w = img.shape
            canvas = torch.zeros((c, max_h, max_w), dtype=img.dtype)
            canvas[:, :h, :w] = img
            padded_images.append(canvas)

        return torch.stack(padded_images), labels

def detr_eval_helper():
    print("Running DETR evaluation (FINAL)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": 512, "longest_edge": 512}
    )

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/val2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_val2017.json",
        #processor=processor
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=detr_CollateFn(),
        num_workers=0
    )

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_val2017.json")

    # CRITICAL FIX → match training
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=91
    )

    model.load_state_dict(torch.load("outputs/detr.pth", map_location=device))
    model.to(device)

    evaluate_model(model, loader, coco_gt, device, processor)

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
        model_path = ""
        if second_choice == "1":
            print("YOLOv8 Training!")
            results = train_yolo(CONFIG)
            model_path = f"{results.save_dir}/weights/best.pt"
            print("Using model from:", model_path)
        elif second_choice == "2":
            if os.path.exists("runs/detect/outputs/yolov8_exp/weights/best.pt"):
                model_path = "runs/detect/outputs/yolov8_exp/weights/best.pt"
                print("YOLOv8 Evaluating!")
                metrics = val_yolo(model_path)
                save_metrics(metrics, "yolov8")
                pred_yolo(model_path, "data/coco_yolo/images/val")
                print("Metrics save to results/yolov8_metrics.json")
            else:
                print("YOLOv8 Model Path Missing, Pls Train The First!")
                sys.exit(130)
        else:
            sys.exit(0)
    
    elif first_choice == "2":
        print("Faster R-CNN Pipeline Triggered!")
        if second_choice == "1":
            print("Faster R-CNN Training!")
            train_frcnn(CONFIG)
        elif second_choice == "2":
            print("Faster R-CNN Evaluating!")
            frcnn_eval_helper()
        else:
            sys.exit(0)
    
    elif first_choice == "3":
        print("DETR")
        if second_choice == "1":
            print("DETR Training!")
            train_detr(CONFIG)
        elif second_choice == "2":
            print("DETR Evaluating!")
            detr_eval_helper()
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

        elif first_choice == "2":
            CONFIG["experiment_name"] = "faster_rcnn"
            show_L2_menu()
            second_choice = input("Enter your model operation (1-3): ").strip()
            main_runner(first_choice, second_choice)

        elif first_choice == "3":
            CONFIG["experiment_name"] = "detr"
            show_L2_menu()
            second_choice = input("Enter your model operation (1-3): ").strip()
            main_runner(first_choice, second_choice)

        elif first_choice == "4":
            print("\nExiting...")
            sys.exit(0)

        else:
            print("\nInvalid choice, try again!\n")


if __name__ == "__main__":
    main()