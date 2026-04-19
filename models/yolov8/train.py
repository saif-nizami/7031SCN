import torch

# FIX PyTorch 2.6 issue globally
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


from ultralytics import YOLO
import random
import numpy as np

def get_device():
    return 0
    # if torch.cuda.is_available():
    #     return 0
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return "mps"   # Apple Silicon GPU
    # else:
    #     return "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(config):
    print("Training YOLOv8...")

    set_seed(config["seed"])

    model = YOLO(config["model_name"])

    model.info()

    device = get_device()
    print(f"Using device: {device}")

    results = model.train(
        data=config["data_yaml"],
        epochs=config["epochs"],
        imgsz=config["img_size"],
        batch=config["batch_size"],
        device=device, #config["device"],
        project=config["project"],
        name=config["experiment_name"],
        exist_ok=config["exist_ok"],
        verbose=True
    )

    print("Training complete!")

    return results