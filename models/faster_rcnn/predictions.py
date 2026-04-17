import torch
import torchvision
import torchvision.transforms as T
import cv2
import os
from tqdm import tqdm

from dataset import COCODataset  # adjust import if needed


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


def get_transform():
    return T.Compose([
        T.ToTensor()
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/val2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_val2017.json",
        transforms=get_transform()
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    num_classes = len(dataset.cat_ids)

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(torch.load("outputs/faster_rcnn.pth"))
    model.to(device)
    model.eval()

    os.makedirs("outputs/vis", exist_ok=True)

    count = 0

    with torch.no_grad():
        for images, targets in tqdm(loader):

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype("uint8").copy()

                boxes = outputs[i]["boxes"].cpu()
                scores = outputs[i]["scores"].cpu()

                for box, score in zip(boxes, scores):
                    if score < 0.5:
                        continue

                    x1, y1, x2, y2 = map(int, box.tolist())

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imwrite(f"outputs/vis/pred_{count}.jpg", img)
                count += 1

                if count >= 5:
                    return


if __name__ == "__main__":
    main()