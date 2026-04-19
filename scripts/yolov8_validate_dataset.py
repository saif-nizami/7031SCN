import os
from pathlib import Path


def validate_split(image_dir, label_dir):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    valid_count = 0
    removed_images = 0
    removed_labels = 0

    print(f"\nValidating: {image_dir}")

    images = list(image_dir.glob("*.jpg"))

    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")

        # Case 1: Label missing
        if not label_path.exists():
            img_path.unlink()
            removed_images += 1
            continue

        # Read label file
        try:
            with open(label_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except:
            # Corrupt file
            img_path.unlink()
            label_path.unlink(missing_ok=True)
            removed_images += 1
            removed_labels += 1
            continue

        # Case 2: Empty label file
        if len(lines) == 0:
            img_path.unlink()
            label_path.unlink()
            removed_images += 1
            removed_labels += 1
            continue

        valid = True

        for line in lines:
            parts = line.split()

            # Wrong format
            if len(parts) != 5:
                valid = False
                break

            try:
                cls, x, y, w, h = map(float, parts)

                # Invalid values
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    valid = False
                    break

                if w <= 0 or h <= 0:
                    valid = False
                    break

            except:
                valid = False
                break

        # Case 3: Invalid label content
        if not valid:
            img_path.unlink()
            label_path.unlink(missing_ok=True)
            removed_images += 1
            removed_labels += 1
        else:
            valid_count += 1

    print(f"Valid images: {valid_count}")
    print(f"Removed images: {removed_images}")
    print(f"Removed labels: {removed_labels}")


def remove_orphan_labels(image_dir, label_dir):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    removed = 0

    print(f"\n🧹 Removing orphan labels in: {label_dir}")

    for label_path in label_dir.glob("*.txt"):
        img_path = image_dir / (label_path.stem + ".jpg")

        if not img_path.exists():
            label_path.unlink()
            removed += 1

    print(f"Removed orphan labels: {removed}")


def main():
    print("\nStarting dataset validation...\n")

    # Train split
    validate_split(
        image_dir="data/coco_yolo/images/train",
        label_dir="data/coco_yolo/labels/train"
    )

    remove_orphan_labels(
        image_dir="data/coco_yolo/images/train",
        label_dir="data/coco_yolo/labels/train"
    )

    # Validation split
    validate_split(
        image_dir="data/coco_yolo/images/val",
        label_dir="data/coco_yolo/labels/val"
    )

    remove_orphan_labels(
        image_dir="data/coco_yolo/images/val",
        label_dir="data/coco_yolo/labels/val"
    )

    print("\nDataset validation COMPLETE!\n")


if __name__ == "__main__":
    main()