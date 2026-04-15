import torch
import os
import cv2
import numpy as np
from PIL import Image

from transformers import DetrForObjectDetection, DetrImageProcessor


def load_model(path, device):
    num_classes = 80  # 🔥 MUST match training

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )

    model.to(device)
    model.eval()

    return model


def predict(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.1
    )[0]

    return results["boxes"]


def draw(image, boxes):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 Load model + processor
    model = load_model("outputs/detr.pth", device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # 🔥 Input/output folders
    input_dir = "data/coco_faster_rcnn_subset/train2017"
    output_dir = "outputs/detr"
    os.makedirs(output_dir, exist_ok=True)

    # 🔥 Limit for speed (change as needed)
    image_files = os.listdir(input_dir)[:10]

    print(f"Processing {len(image_files)} images...\n")

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")

            boxes = predict(model, processor, image, device)

            result = draw(image, boxes)

            cv2.imwrite(os.path.join(output_dir, img_name), result)

            print(f"[{i+1}/{len(image_files)}] Done: {img_name}")

        except Exception as e:
            print(f"Error with {img_name}: {e}")

    print("\n✅ DETR inference completed!")