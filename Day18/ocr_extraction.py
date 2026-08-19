import easyocr
import cv2
import json
import os
from pathlib import Path


# ============================================================
# DAY 18 - EASY OCR
# ============================================================

print("=" * 60)
print("DAY 18 - EASY OCR TEXT EXTRACTION")
print("=" * 60)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\OCR_YOLO_Training")

DATASET_ROOT = PROJECT_ROOT / "Day16" / "dataset"

TRAIN_IMAGES = DATASET_ROOT / "images" / "train"
VAL_IMAGES = DATASET_ROOT / "images" / "val"

TRAIN_LABELS = DATASET_ROOT / "labels" / "train"
VAL_LABELS = DATASET_ROOT / "labels" / "val"

RESULT_DIR = PROJECT_ROOT / "Day18" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = RESULT_DIR / "ocr_results.json"


# ------------------------------------------------------------
# CREATE OCR READER
# ------------------------------------------------------------

print("\nLoading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded successfully.")


# ------------------------------------------------------------
# FUNCTION TO READ YOLO LABEL
# ------------------------------------------------------------

def read_yolo_label(label_path):

    with open(label_path, "r") as file:
        line = file.readline().strip()

    if not line:
        return None

    parts = line.split()

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    return class_id, x_center, y_center, width, height


# ------------------------------------------------------------
# FUNCTION TO CROP PLATE
# ------------------------------------------------------------

def crop_plate(image, label):

    class_id, x_center, y_center, width, height = label

    img_height, img_width = image.shape[:2]

    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)

    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    crop = image[y1:y2, x1:x2]

    return crop


# ------------------------------------------------------------
# GET ALL IMAGES
# ------------------------------------------------------------

image_label_pairs = []

for image_dir, label_dir in [
    (TRAIN_IMAGES, TRAIN_LABELS),
    (VAL_IMAGES, VAL_LABELS)
]:

    for image_path in sorted(image_dir.glob("*.jpg")):

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            image_label_pairs.append(
                (image_path, label_path)
            )


print("\nTotal labelled images found:", len(image_label_pairs))


# ------------------------------------------------------------
# PROCESS IMAGES
# ------------------------------------------------------------

results = []

for index, (image_path, label_path) in enumerate(
    image_label_pairs,
    start=1
):

    print("\n" + "-" * 60)
    print(f"Processing {index}/{len(image_label_pairs)}")
    print("Image:", image_path.name)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        continue

    label = read_yolo_label(label_path)

    if label is None:
        print("Invalid label.")
        continue

    plate_crop = crop_plate(image, label)

    if plate_crop.size == 0:
        print("Empty plate crop.")
        continue

    # --------------------------------------------------------
    # EASY OCR
    # --------------------------------------------------------

    ocr_output = reader.readtext(
        plate_crop,
        detail=1,
        paragraph=False
    )

    image_results = []

    for detection in ocr_output:

        bbox, text, confidence = detection

    # Convert numpy values to normal Python values
    clean_bbox = [
        [int(point[0]), int(point[1])]
        for point in bbox
    ]

    image_results.append({
        "bbox": clean_bbox,
        "text": str(text),
        "confidence": float(confidence)
    })

    print(
        f"Text: {text} | "
        f"Confidence: {confidence:.4f}"
    )
    print(
            f"Text: {text} | "
            f"Confidence: {confidence:.4f}"
        )

    results.append({
        "image": image_path.name,
        "label": label_path.name,
        "ocr_results": image_results
    })


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OCR PROCESS COMPLETED")
print("=" * 60)

print("Images processed :", len(results))
print("JSON saved to    :", OUTPUT_JSON)

print("=" * 60)