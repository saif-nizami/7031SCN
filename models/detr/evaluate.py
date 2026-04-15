import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from transformers import DetrForObjectDetection, DetrImageProcessor
from models.faster_rcnn.dataset import COCODataset

# ---------------- IOU ----------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (a1 + a2 - inter + 1e-6)


# ---------------- EVALUATION ----------------
def evaluate(model, loader, device):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model.eval()

    TP, FP, FN = 0, 0, 0

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = [img.to(device) for img in images]

            # DETR expects batch tensor
            pixel_values = torch.stack(images)

            outputs = model(pixel_values=pixel_values)

            # convert outputs → boxes
            target_sizes = torch.tensor(
                [img.shape[-2:] for img in images]
            ).to(device)

            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.1  # 🔥 lower threshold
            )

            for pred, target in zip(results, targets):
                pb = pred["boxes"].cpu()
                tb = target["boxes"].cpu()

                matched = set()

                for p in pb:
                    found = False
                    for i, t in enumerate(tb):
                        if i in matched:
                            continue
                        if compute_iou(p, t) > 0.5:
                            TP += 1
                            matched.add(i)
                            found = True
                            break
                    if not found:
                        FP += 1

                FN += len(tb) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return precision, recall

# ------------ EVALUATE COCO -----------
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def evaluate_coco(model, loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        count = 0
        for images, targets in list(loader)[:50]:
            count = count + 1
            print("count :", count)
            images = [img.to(device) for img in images]
            pixel_values = torch.stack(images)

            outputs = model(pixel_values=pixel_values)

            # convert outputs
            target_sizes = torch.tensor(
                [img.shape[-2:] for img in images]
            ).to(device)

            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

            preds = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.1
            )

            for pred, target in zip(preds, targets):
                image_id = int(target.get("image_id", 0))

                for box, score, label in zip(
                    pred["boxes"], pred["scores"], pred["labels"]
                ):
                    x1, y1, x2, y2 = box.tolist()

                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(score)
                    })

    with open("detr_preds.json", "w") as f:
        json.dump(results, f)

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_train2017.json")
    coco_dt = coco_gt.loadRes("detr_preds.json")

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🚀 Running DETR evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    def get_transform():
        return T.Compose([T.ToTensor()])

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        transforms=get_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 🔥 USE PRETRAINED DETR (recommended)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    ).to(device)

    # OR if you insist on your trained model:
    # model = DetrForObjectDetection.from_pretrained(
    #     "facebook/detr-resnet-50",
    #     num_labels=80,
    #     ignore_mismatched_sizes=True
    # ).to(device)
    # model.load_state_dict(torch.load("outputs/detr.pth", map_location=device, weights_only=True))

    # evaluate(model, loader, device)
    evaluate_coco(model, loader, device)