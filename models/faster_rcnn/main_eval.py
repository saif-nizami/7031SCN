import torch
import time
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import torchvision

# from models.faster_rcnn.dataset import COCODataset
from dataset import COCODataset

import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.ToTensor()
    ])


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


def evaluate_model(model, dataloader, coco_gt, device, label2cat):
    model.eval()

    results = []
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader):

            if len(images) == 0:
                continue

            images = [img.to(device) for img in images]

            start = time.time()
            outputs = model(images)
            end = time.time()

            total_time += (end - start)
            total_images += len(images)

            for i, output in enumerate(outputs):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                img_id = int(targets[i]["image_id"].item())

                for box, score, label in zip(boxes, scores, labels):

                    # skip very low confidence (optional but recommended)
                    if score < 0.05:
                        continue

                    if not torch.isfinite(box).all():
                        continue

                    x1, y1, x2, y2 = box.tolist()

                    w = x2 - x1
                    h = y2 - y1

                    if w <= 0 or h <= 0:
                        continue

                    results.append({
                        "image_id": img_id,
                        "category_id": int(label2cat[int(label)]),  # 🔥 FIXED
                        "bbox": [x1, y1, w, h],
                        "score": float(score)
                    })

    if len(results) == 0:
        print("❌ No valid predictions found!")
        return {}

    # COCO evaluation
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    mAP50_95 = coco_eval.stats[0]
    mAP50 = coco_eval.stats[1]

    # Approx precision & recall
    precision = coco_eval.stats[1]
    recall = coco_eval.stats[8]

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Inference
    avg_time = total_time / total_images
    fps = 1 / avg_time

    print("\nFinal Metrics:")
    print(f"mAP@50: {mAP50:.4f}")
    print(f"mAP@50-95: {mAP50_95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Inference Time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "inference_time_ms": avg_time * 1000,
        "fps": fps
    }


# ✅ MAIN BLOCK (FIXED)
if __name__ == "__main__":

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
        collate_fn=collate_fn,
        num_workers=2
    )

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_val2017.json")

    # 🔥 rebuild model
    num_classes = len(dataset.cat_ids)

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(torch.load("outputs/faster_rcnn.pth"))
    model.to(device)

    # 🔥 mapping back to COCO category IDs
    label2cat = {v: k for k, v in dataset.cat2label.items()}

    metrics = evaluate_model(model, val_loader, coco_gt, device, label2cat)