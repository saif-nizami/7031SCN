from ultralytics import YOLO

def validate_model(model_path):
    print("Validating model...")

    model = YOLO(model_path)

    metrics = model.val()

    # Extract base metrics
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Inference time
    inference_time = metrics.speed['inference']  # ms/image
    fps = 1000 / inference_time if inference_time > 0 else 0

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "inference_time_ms": inference_time,
        "fps": fps
    }

    print("\nMetrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results