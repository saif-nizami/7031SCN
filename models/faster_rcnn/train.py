import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.faster_rcnn.dataset import COCODataset


def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # remove invalid samples
    return tuple(zip(*batch))

def train_model(config):
    print("🚀 Training Faster R-CNN...")

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        transforms=get_transform()
    )

    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print("Selected Device: ", device)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    count = 0
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}")
    
        for images, targets in data_loader:
            count = count + 1
            print('count : ', count)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print("Loss:", losses.item())

    torch.save(model.state_dict(), "outputs/faster_rcnn.pth")

    print("✅ Training complete!")

    return "outputs/faster_rcnn.pth"