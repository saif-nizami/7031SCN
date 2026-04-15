import json
import random
import os
import shutil

SOURCE = "data/coco_raw"
DEST = "data/coco_faster_rcnn_subset"

TRAIN_SIZE = 2000
VAL_SIZE = 500

random.seed(42)


def create_subset(split, size):
    print(f"\n🔧 Processing {split}...")

    img_dir = f"{SOURCE}/{split}2017"
    ann_file = f"{SOURCE}/annotations/instances_{split}2017.json"

    with open(ann_file) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    selected_images = random.sample(images, min(size, len(images)))
    selected_ids = {img["id"] for img in selected_images}

    selected_annotations = [
        ann for ann in annotations if ann["image_id"] in selected_ids
    ]

    # Create dirs
    os.makedirs(f"{DEST}/{split}2017", exist_ok=True)
    os.makedirs(f"{DEST}/annotations", exist_ok=True)

    # Copy images
    for img in selected_images:
        src = os.path.join(img_dir, img["file_name"])
        dst = os.path.join(f"{DEST}/{split}2017", img["file_name"])
        shutil.copy(src, dst)

    # Save new JSON
    new_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco["categories"]
    }

    with open(f"{DEST}/annotations/instances_{split}2017.json", "w") as f:
        json.dump(new_coco, f)

    print(f"✅ {split}: {len(selected_images)} images")


def main():
    print("🚀 Creating COCO subset...")

    create_subset("train", TRAIN_SIZE)
    create_subset("val", VAL_SIZE)

    print("\n🎉 Subset ready at:", DEST)


if __name__ == "__main__":
    main()