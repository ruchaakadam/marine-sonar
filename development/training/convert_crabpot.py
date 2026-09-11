import json
from pathlib import Path
from PIL import Image

# ============================================================
# CRAB-POT DATASET → YOLO FORMAT
# ============================================================

SOURCE = Path(r"C:\Users\Aarya\Downloads\sss-crab-pot-detection-ds")

OUTPUT = Path(r"C:\Users\Aarya\marine-sonar\yolo_dataset")

CLASS_ID = 0  # crab_pot


def convert_split(split):
    image_dir = SOURCE / split
    metadata_file = image_dir / "metadata.jsonl"

    output_images = OUTPUT / split / "images"
    output_labels = OUTPUT / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    if not metadata_file.exists():
        print(f"Skipping {split}: metadata.jsonl not found")
        return

    count = 0

    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            image_name = item["file_name"]
            image_path = image_dir / image_name

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue

            # Copy image
            output_image = output_images / image_name
            output_image.write_bytes(image_path.read_bytes())

            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

                objects = item.get("objects", {})
                labels = objects.get("bbox", [])

            output_label = output_labels / (Path(image_name).stem + ".txt")

            with open(output_label, "w", encoding="utf-8") as lf:
                for box in labels:
                    x, y, w, h = box

                    # Convert XYWH pixels → YOLO normalized XYWH
                    center_x = (x + w / 2) / width
                    center_y = (y + h / 2) / height
                    norm_w = w / width
                    norm_h = h / height

                    lf.write(
                        f"{CLASS_ID} "
                        f"{center_x:.6f} "
                        f"{center_y:.6f} "
                        f"{norm_w:.6f} "
                        f"{norm_h:.6f}\n"
                    )

            count += 1

    print(f"{split}: converted {count} images")


for split in ["train", "valid", "test"]:
    convert_split(split)

print("\nConversion complete.")
print(f"YOLO dataset created at: {OUTPUT}")