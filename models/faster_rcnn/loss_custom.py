import json
import matplotlib.pyplot as plt
import numpy as np

# ---- Load losses.json ----
with open("losses.json", "r") as f:
    losses = json.load(f)

print(f"Loaded {len(losses)} loss values")

# ---- Create moving average (smoothing) ----
def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_losses = moving_average(losses, window_size=50)

# ---- Plot ----
plt.figure()

# Raw loss (light)
plt.plot(losses, alpha=0.4, label="Raw Loss")

# Smoothed loss (main line)
plt.plot(range(len(smoothed_losses)), smoothed_losses, linewidth=2, label="Smoothed Loss")

plt.title("Faster R-CNN Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

# ---- Save + Show ----
plt.savefig("frcnn_loss_plot.png")
plt.show()

print("Plot saved as frcnn_loss_plot.png")