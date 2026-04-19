import os
import json
import torch
from PIL import Image


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, processor):
        self.root = root
        self.processor = processor

        with open(annotation) as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}

        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        self.ids = [
            img_id for img in self.images
            if img in self.annotations and len(self.annotations[img]) > 0
        ]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations[img_id]

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            if w <= 1 or h <= 1:
                continue

            boxes.append([x, y, w, h])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            return None

        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": box[2] * box[3],
                    "iscrowd": 0
                }
                for box, label in zip(boxes, labels)
            ]
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels

    def __len__(self):
        return len(self.ids)