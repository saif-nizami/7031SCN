import os
import torch
from PIL import Image
import json


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(annotation) as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}

        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = [
            img_id for img_id in self.images.keys()
            if img_id in self.annotations and len(self.annotations[img_id]) > 0
        ]

        self.cat_ids = sorted({ann["category_id"] for ann in self.coco["annotations"]})
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        width, height = img.size

        anns = self.annotations[img_id]

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # skip invalid boxes
            if w <= 1 or h <= 1:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[ann["category_id"]])

        # skip bad samples completely
        if len(boxes) == 0:
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # FINAL SAFETY
        if not torch.isfinite(boxes).all():
            return None

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])

        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)