import json
import os


def save_metrics(metrics, model_name):
    os.makedirs("results", exist_ok=True)

    path = f"results/{model_name}_metrics.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to: {path}")