import os
import random
import shutil
from pathlib import Path

# CONFIG
SOURCE = "data/coco_yolo"
DEST = "data/coco_yolo_subset"
TRAIN_SIZE = 2000
VAL_SIZE = 500

random.seed(42)


def create_subset(split, size):
    src_img = Path(f"{SOURCE}/images/{split}")
    src_lbl = Path(f"{SOURCE}/labels/{split}")

    dst_img = Path(f"{DEST}/images/{split}")
    dst_lbl = Path(f"{DEST}/labels/{split}")

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = list(src_img.glob("*.jpg"))

    # Only keep images that HAVE labels
    images = [img for img in images if (src_lbl / (img.stem + ".txt")).exists()]

    selected = random.sample(images, min(size, len(images)))

    for img_path in selected:
        lbl_path = src_lbl / (img_path.stem + ".txt")

        shutil.copy(img_path, dst_img / img_path.name)
        shutil.copy(lbl_path, dst_lbl / lbl_path.name)

    print(f"{split}: {len(selected)} images copied")


def main():
    print("Creating subset dataset...\n")

    create_subset("train", TRAIN_SIZE)
    create_subset("val", VAL_SIZE)

    print("\nSubset created at:", DEST)


if __name__ == "__main__":
    main()