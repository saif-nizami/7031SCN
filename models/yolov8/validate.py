from ultralytics import YOLO

# def validate_model(model_path):
#     print("📊 Validating model...")

#     model = YOLO(model_path)

#     metrics = model.val()

#     print("mAP50:", metrics.box.map50)
#     print("mAP50-95:", metrics.box.map)
#     print("Precision:", metrics.box.mp)
#     print("Recall:", metrics.box.mr)

#     return metrics

def validate_model(model_path):
    print("📊 Validating model...")

    model = YOLO(model_path)

    metrics = model.val()

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr)
    }

    print("\n📈 Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results