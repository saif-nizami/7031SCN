import torch
import time
from torch.utils.data import DataLoader
import torchvision.transforms as T

from transformers import DetrForObjectDetection, DetrImageProcessor
from models.faster_rcnn.dataset import COCODataset


# ---------------- BENCHMARK ----------------
def benchmark(model, loader, device):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model.eval()

    total_time = 0
    total_images = 0

    with torch.no_grad():
        for images, _ in loader:
            images = [img.to(device) for img in images]

            pixel_values = torch.stack(images)

            start = time.time()
            outputs = model(pixel_values=pixel_values)
            end = time.time()

            total_time += (end - start)
            total_images += len(images)

    print("\nAvg time per image:", total_time / total_images)
    print("FPS:", total_images / total_time)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🚀 Running DETR benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    def get_transform():
        return T.Compose([T.ToTensor()])

    dataset = COCODataset(
        root="data/coco_faster_rcnn_subset/train2017",
        annotation="data/coco_faster_rcnn_subset/annotations/instances_train2017.json",
        transforms=get_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 🔥 Use pretrained DETR (recommended)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    ).to(device)

    # Optional: limit dataset for speed
    loader = list(loader)[:50]

    benchmark(model, loader, device)