import torch
from tqdm import tqdm

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate(model, dataloader, device, iou_thresh=0.5):
    model.eval()

    TP, FP, FN = 0, 0, 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]

            outputs = model(images)

            for pred, target in zip(outputs, targets):
                pred_boxes = pred["boxes"].cpu()
                true_boxes = target["boxes"].cpu()

                matched = set()

                for pb in pred_boxes:
                    found = False
                    for i, tb in enumerate(true_boxes):
                        if i in matched:
                            continue
                        if compute_iou(pb, tb) > iou_thresh:
                            TP += 1
                            matched.add(i)
                            found = True
                            break
                    if not found:
                        FP += 1

                FN += len(true_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return precision, recall

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
    evaluate(model, data_loader, device)