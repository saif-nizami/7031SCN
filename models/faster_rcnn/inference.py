import torch
from PIL import Image
import torchvision.transforms as T
import cv2
import os

from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_model(weights_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device, threshold=0.5):
    image = Image.open(image_path).convert("RGB")

    transform = T.ToTensor()
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()

    # Filter
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    return image, boxes, labels, scores


def draw_boxes(image, boxes):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


if __name__ == "__main__":
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model("outputs/faster_rcnn.pth", device)

    input_dir = "data/coco_faster_rcnn_subset/train2017"
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for img_name in os.listdir(input_dir):
        path = os.path.join(input_dir, img_name)
        count = count + 1
        print("count :", count)
        image, boxes, labels, scores = predict(model, path, device)

        result = draw_boxes(image, boxes)

        cv2.imwrite(os.path.join(output_dir, img_name), result)

    print("✅ Inference complete!")