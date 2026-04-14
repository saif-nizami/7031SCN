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

        # Image dict
        self.images = {img["id"]: img for img in self.coco["images"]}

        # Annotation dict
        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # ✅ FIX 1: Filter out images with no annotations
        self.ids = []
        for img_id in self.images.keys():
            if img_id in self.annotations and len(self.annotations[img_id]) > 0:
                self.ids.append(img_id)

        # ✅ FIX 2: Create category mapping (COCO IDs → 0...N-1)
        self.cat_ids = sorted({ann["category_id"] for ann in self.coco["annotations"]})
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.annotations[img_id]

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # Convert COCO → [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])

            # Remap category IDs
            labels.append(self.cat2label[ann["category_id"]])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# import os
# import torch
# from PIL import Image
# import json


# class COCODataset(torch.utils.data.Dataset):
#     def __init__(self, root, annotation, transforms=None):
#         self.root = root
#         self.transforms = transforms

#         with open(annotation) as f:
#             self.coco = json.load(f)

#         self.images = {img["id"]: img for img in self.coco["images"]}

#         self.annotations = {}
#         for ann in self.coco["annotations"]:
#             img_id = ann["image_id"]
#             if img_id not in self.annotations:
#                 self.annotations[img_id] = []
#             self.annotations[img_id].append(ann)

#         self.ids = list(self.images.keys())

#     def __getitem__(self, idx):
#         img_id = self.ids[idx]
#         img_info = self.images[img_id]

#         img_path = os.path.join(self.root, img_info["file_name"])
#         img = Image.open(img_path).convert("RGB")

#         anns = self.annotations.get(img_id, [])

#         boxes = []
#         labels = []

#         for ann in anns:
#             x, y, w, h = ann["bbox"]
#             boxes.append([x, y, x + w, y + h])
#             labels.append(ann["category_id"])

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)

#         target = {
#             "boxes": boxes,
#             "labels": labels
#         }

#         if self.transforms:
#             img = self.transforms(img)

#         return img, target

#     def __len__(self):
#         return len(self.ids)