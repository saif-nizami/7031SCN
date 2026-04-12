import os

image_dir = "data/coco_yolo/images/train"
label_dir = "data/coco_yolo/labels/train"

images = os.listdir(image_dir)

removed = 0

for img in images:
    label_name = img.replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(label_path):
        os.remove(os.path.join(image_dir, img))
        removed += 1

print(f"✅ Removed {removed} images without labels")