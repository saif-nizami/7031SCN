import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import torchvision
import torchvision.transforms as T

from dataset import COCODataset  # adjust path


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

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_val2017.json")

    num_classes = len(dataset.cat_ids)

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(torch.load("outputs/faster_rcnn.pth"))
    model.to(device)
    model.eval()

    label2cat = {v: k for k, v in dataset.cat2label.items()}

    results = []

    with torch.no_grad():
        for images, targets in tqdm(loader):

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                img_id = int(targets[i]["image_id"])

                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.05:
                        continue

                    x1, y1, x2, y2 = box.tolist()

                    results.append({
                        "image_id": img_id,
                        "category_id": int(label2cat[int(label)]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    precisions = coco_eval.eval['precision']

    pr = precisions[0, :, :, 0, 2]  # IoU=0.5
    pr = pr.mean(axis=1)

    recall = np.linspace(0, 1, len(pr))

    plt.figure()
    plt.plot(recall, pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Faster R-CNN)")
    plt.savefig("outputs/pr_curve.png")
    plt.show()


if __name__ == "__main__":
    main()