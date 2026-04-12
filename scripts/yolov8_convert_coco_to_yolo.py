from ultralytics.data.converter import convert_coco

def main():
    convert_coco(
        labels_dir="data/coco_raw/annotations",
        save_dir="data/coco_yolo"
    )

if __name__ == "__main__":
    main()