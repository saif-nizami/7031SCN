from ultralytics import YOLO


def run_inference(model_path, source):
    print("Running inference...")

    model = YOLO(model_path)

    results = model.predict(
        source=source,
        save=True,
        conf=0.25
    )

    print("Inference complete!")

    return results