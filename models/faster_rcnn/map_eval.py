from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json

def evaluate_coco(model, dataloader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in list(dataloader)[:50]:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"]) if "image_id" in target else 0

                for box, score, label in zip(
                    output["boxes"], output["scores"], output["labels"]
                ):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(score)
                    })

    with open("predictions.json", "w") as f:
        json.dump(results, f)

    coco_gt = COCO("data/coco_faster_rcnn_subset/annotations/instances_train2017.json")
    coco_dt = coco_gt.loadRes("predictions.json")

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    # from models.faster_rcnn.dataset import COCODataset
    from dataset import COCODataset

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    # dataset = COCODataset(
    #     root="data/coco_faster_rcnn_subset/train2017",
    #     annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
    #     transforms=None
    # )
    def get_transform():
        return T.Compose([T.ToTensor()])

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        transforms=get_transform()
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # model
    model = fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load("outputs/faster_rcnn.pth", map_location=device))
    model.to(device)

    # run evaluation
    evaluate_coco(model, data_loader, device)