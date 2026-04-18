import torch
import time
from tqdm import tqdm

from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset_detr import COCODataset


# =========================
# COLLATE (manual padding)
# =========================
class CollateFn:
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


# =========================
# EVALUATION
# =========================
def evaluate_model(model, dataloader, coco_gt, device, processor):
    model.eval()

    results = []
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for pixel_values, targets in tqdm(dataloader):

            if len(pixel_values) == 0:
                continue

            pixel_values = pixel_values.to(device)

            start = time.time()
            outputs = model(pixel_values=pixel_values)
            end = time.time()

            total_time += (end - start)
            total_images += pixel_values.shape[0]

            processed = processor.post_process_object_detection(
                outputs,
                threshold=0.1,  # 🔥 important for DETR
                target_sizes=[(512, 512)] * pixel_values.shape[0]
            )

            for i, p in enumerate(processed):
                img_id = int(targets[i]["image_id"])

                # DEBUG
                if i == 0:
                    print("Predictions in first image:", len(p["boxes"]))

                for box, score, label in zip(p["boxes"], p["scores"], p["labels"]):
                    x1, y1, x2, y2 = box.tolist()

                    results.append({
                        "image_id": img_id,
                        "category_id": int(label),  # ✅ correct (no remap)
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    if len(results) == 0:
        print("❌ No detections → model likely undertrained")
        return

    # =========================
    # COCO EVAL
    # =========================
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP50_95 = coco_eval.stats[0]
    mAP50 = coco_eval.stats[1]
    recall = coco_eval.stats[8]

    precision = mAP50
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    avg_time = total_time / total_images
    fps = 1 / avg_time

    print("\nFinal Metrics:")
    print(f"mAP50: {mAP50:.4f}")
    print(f"mAP50-95: {mAP50_95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"FPS: {fps:.2f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("🚀 Running DETR evaluation (FINAL)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": 512, "longest_edge": 512}
    )

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/val2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_val2017.json",
        processor=processor
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CollateFn(),
        num_workers=0
    )

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_val2017.json")

    # 🔥 CRITICAL FIX → match training
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=91  # 🔥 MUST match your checkpoint
    )

    model.load_state_dict(torch.load("outputs/detr.pth", map_location=device))
    model.to(device)

    evaluate_model(model, loader, coco_gt, device, processor)