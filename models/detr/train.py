import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from transformers import DetrForObjectDetection, DetrImageProcessor
from models.detr.dataset_detr import COCODataset


# =========================
# COLLATE
# =========================
class CollateFn:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # manual padding
        max_h = max(img.shape[1] for img in pixel_values)
        max_w = max(img.shape[2] for img in pixel_values)

        padded_images = []

        for img in pixel_values:
            c, h, w = img.shape
            padded = torch.zeros((c, max_h, max_w), dtype=img.dtype)
            padded[:, :h, :w] = img
            padded_images.append(padded)

        pixel_values = torch.stack(padded_images)

        return pixel_values, labels


# =========================
# TRAIN
# =========================
def train_model(config):
    print("Training DETR (stable mode)...")

    os.makedirs("outputs", exist_ok=True)

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": 512, "longest_edge": 512}  # speed + stability
    )

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        processor=processor
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=CollateFn(),
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")

        loop = tqdm(loader)

        for pixel_values, labels in loop:

            pixel_values = pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # hard NaN guard
            if not torch.isfinite(loss):
                print("⚠️ Skipping bad batch")
                continue

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping (prevents NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            loop.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), "outputs/detr.pth")

    print("🎉 DETR Training Completed Successfully!")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    config = {
        "epochs": 5
    }

    train_model(config)