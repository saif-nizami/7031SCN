# import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torch.utils.data import DataLoader
# import torchvision.transforms as T

# from models.faster_rcnn.dataset import COCODataset


# def get_transform():
#     return T.Compose([T.ToTensor()])

# def collate_fn(batch):
#     batch = [b for b in batch if b is not None]  # remove invalid samples
#     return tuple(zip(*batch))

# def train_model(config):
#     print("🚀 Training Faster R-CNN...")

#     dataset = COCODataset(
#         root="data/coco_faster_rcnn_subset/train2017",
#         annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
#         transforms=get_transform()
#     )

#     # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
#     data_loader = DataLoader(
#         dataset,
#         batch_size=2,
#         shuffle=True,
#         collate_fn=collate_fn
#     )

#     # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     device = torch.device(
#         "cuda" if torch.cuda.is_available()
#         else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
#         else "cpu"
#     )
#     print("Selected Device: ", device)

#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     model.to(device)

#     model.train()

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
#     count = 0
#     for epoch in range(config["epochs"]):
#         print(f"\nEpoch {epoch+1}")
    
#         for images, targets in data_loader:
#             count = count + 1
#             print('count : ', count)
#             images = [img.to(device) for img in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#         print("Loss:", losses.item())

#     torch.save(model.state_dict(), "outputs/faster_rcnn.pth")

#     print("✅ Training complete!")

#     return "outputs/faster_rcnn.pth"

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import os
import json

from models.faster_rcnn.dataset import COCODataset


def get_transform():
    return T.Compose([
        T.ToTensor()
    ])


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def train_model(config):
    print("🚀 Training Faster R-CNN...")

    os.makedirs("outputs", exist_ok=True)

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        transforms=get_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,       # 🔥 speed boost
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(dataset.cat_ids)

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )

    # replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    # 🔥 NEW: tracking
    losses_list = []
    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        print(f"\n📘 Epoch [{epoch+1}/{config['epochs']}]")

        epoch_loss = 0
        steps = 0

        loop = tqdm(loader)

        for images, targets in loop:

            if len(images) == 0:
                continue

            images = [img.to(device) for img in images]

            clean_targets = []
            for t in targets:
                boxes = t["boxes"]
                labels = t["labels"]

                valid = torch.isfinite(boxes).all(dim=1)

                x1, y1, x2, y2 = boxes.unbind(1)
                valid = valid & (x2 > x1) & (y2 > y1)

                boxes = boxes[valid]
                labels = labels[valid]

                if boxes.shape[0] == 0:
                    continue

                clean_targets.append({
                    "boxes": boxes.to(device),
                    "labels": labels.to(device)
                })

            if len(clean_targets) == 0:
                continue

            loss_dict = model(images, clean_targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print("⚠️ Skipping NaN batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()

            losses_list.append(loss_value)
            epoch_loss += loss_value
            steps += 1

            loop.set_postfix(loss=f"{loss_value:.4f}")

        # 🔥 average epoch loss
        avg_loss = epoch_loss / max(steps, 1)
        print(f"✅ Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # 🔥 save every epoch
        torch.save(
            model.state_dict(),
            f"outputs/faster_rcnn_epoch_{epoch+1}.pth"
        )

        # 🔥 save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "outputs/faster_rcnn_best.pth")

    # ✅ final save
    torch.save(model.state_dict(), "outputs/faster_rcnn.pth")

    # 🔥 save loss list for plotting
    with open("outputs/losses.json", "w") as f:
        json.dump(losses_list, f)

    print("🎉 Training Complete!")
    print("📊 Loss data saved → outputs/losses.json")