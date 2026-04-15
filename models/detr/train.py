import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from transformers import DetrForObjectDetection
from models.faster_rcnn.dataset import COCODataset


def get_transform():
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])


def train_model(config):
    print("🚀 Training DETR...")

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        transforms=get_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,  # 🔥 safer for DETR + small GPU
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = DetrForObjectDetection.from_pretrained(
    #     "facebook/detr-resnet-50"
    # ).to(device)

    num_classes = len(dataset.cat_ids)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    count = 0
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}")

        for images, targets in loader:
            count = count + 1
            print("count :", count)
            # move images to device
            images = [img.to(device) for img in images]

            formatted_targets = []
            valid_images = []

            for img, t in zip(images, targets):
                boxes = t["boxes"].clone().float()
                labels = t["labels"].clone()

                # skip empty
                if boxes.numel() == 0:
                    continue

                # ensure valid
                x1, y1, x2, y2 = boxes.unbind(1)

                w = (x2 - x1).clamp(min=1)
                h = (y2 - y1).clamp(min=1)

                cx = x1 + w / 2
                cy = y1 + h / 2

                boxes = torch.stack([cx, cy, w, h], dim=1)

                # normalize (IMPORTANT)
                boxes = boxes / 512.0

                # remove bad
                valid = (w > 1) & (h > 1)
                boxes = boxes[valid]
                labels = labels[valid]

                if boxes.numel() == 0:
                    continue

                formatted_targets.append({
                    "class_labels": labels.to(device),
                    "boxes": boxes.to(device)
                })
                valid_images.append(img)

            # skip batch if nothing valid
            if len(formatted_targets) == 0:
                continue

            pixel_values = torch.stack(valid_images)

            outputs = model(
                pixel_values=pixel_values,
                labels=formatted_targets
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss:", loss.item())

    torch.save(model.state_dict(), "outputs/detr.pth")

    print("✅ DETR Training complete!")

    return "outputs/detr.pth"