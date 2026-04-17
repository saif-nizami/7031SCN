import json
import matplotlib.pyplot as plt


def main():
    with open("outputs/losses.json") as f:
        losses = json.load(f)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Faster R-CNN)")
    plt.savefig("outputs/faster_rcnn_loss.png")
    plt.show()


if __name__ == "__main__":
    main()