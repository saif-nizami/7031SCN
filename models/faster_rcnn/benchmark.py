import torch
import time

def benchmark(model, dataloader, device):
    model.eval()

    total_time = 0
    total_images = 0

    with torch.no_grad():
        for images, _ in list(dataloader)[:50]:
            images = [img.to(device) for img in images]

            start = time.time()
            _ = model(images)
            end = time.time()

            total_time += (end - start)
            total_images += len(images)

    fps = total_images / total_time

    print(f"Avg time per image: {total_time / total_images:.4f} sec")
    print(f"FPS: {fps:.2f}")

    return fps

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    import torchvision.transforms as T

    # from models.faster_rcnn.dataset import COCODataset
    from dataset import COCODataset


    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms (IMPORTANT — same fix as evaluate)
    def get_transform():
        return T.Compose([T.ToTensor()])

    # dataset
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

    # run benchmark
    benchmark(model, data_loader, device)