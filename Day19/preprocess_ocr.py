import cv2
import easyocr
import json
import re
from pathlib import Path


# ============================================================
# DAY 19 - EASYOCR + OPENCV
# OCR PREPROCESSING AND POST-PROCESSING
# ============================================================

PROJECT_ROOT = Path(r"C:\OCR_YOLO_Training")

DATASET_ROOT = PROJECT_ROOT / "Day16" / "dataset"

TRAIN_IMAGES = DATASET_ROOT / "images" / "train"
VAL_IMAGES = DATASET_ROOT / "images" / "val"

TRAIN_LABELS = DATASET_ROOT / "labels" / "train"
VAL_LABELS = DATASET_ROOT / "labels" / "val"

DAY19_DIR = PROJECT_ROOT / "Day19"

CROPS_DIR = DAY19_DIR / "crops"
RESULTS_DIR = DAY19_DIR / "results"

CROPS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EASY OCR
# ============================================================

print("=" * 60)
print("DAY 19 - OCR PREPROCESSING AND POST-PROCESSING")
print("=" * 60)

print("\nLoading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded successfully.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_plate_text(text):
    """
    Normalize OCR text for Indian number-plate style text.
    """

    if text is None:
        return ""

    text = str(text).upper()

    # Remove spaces and special characters
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Common OCR character corrections
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    cleaned = ""

    for char in text:
        cleaned += replacements.get(char, char)

    return cleaned


# ============================================================
# INDIAN PLATE FORMAT VALIDATION
# ============================================================

def is_valid_indian_plate(text):
    """
    Basic Indian vehicle registration pattern.

    Example:
    MH12AB1234
    MH20CD5678
    """

    text = clean_plate_text(text)

    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"

    return bool(re.fullmatch(pattern, text))


# ============================================================
# YOLO LABEL READER
# ============================================================

def read_yolo_label(label_path):

    with open(label_path, "r") as file:
        line = file.readline().strip()

    if not line:
        return None

    parts = line.split()

    return (
        int(parts[0]),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4])
    )


# ============================================================
# YOLO CROP
# ============================================================

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

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    crop = image[y1:y2, x1:x2]

    return crop


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def grayscale(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def clahe_enhancement(image):

    gray = grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def threshold_image(image):

    gray = grayscale(image)

    _, thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresholded


# ============================================================
# OCR FUNCTION
# ============================================================

def run_ocr(image):

    output = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    results = []

    for detection in output:

        bbox, text, confidence = detection

        results.append({
            "bbox": [
                [int(point[0]), int(point[1])]
                for point in bbox
            ],
            "raw_text": str(text),
            "cleaned_text": clean_plate_text(text),
            "confidence": float(confidence),
            "valid_indian_plate": is_valid_indian_plate(text)
        })

    return results


# ============================================================
# FIND ALL 20 IMAGES
# ============================================================

image_label_pairs = []

for image_dir, label_dir in [
    (TRAIN_IMAGES, TRAIN_LABELS),
    (VAL_IMAGES, VAL_LABELS)
]:

    for image_path in sorted(
        image_dir.glob("*.jpg")
    ):

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():

            image_label_pairs.append(
                (image_path, label_path)
            )


print(
    f"\nTotal labelled images found: "
    f"{len(image_label_pairs)}"
)


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

all_results = []


for index, (image_path, label_path) in enumerate(
    image_label_pairs,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"Processing {index}/"
        f"{len(image_label_pairs)}"
    )

    print(
        f"Image: {image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("Could not read image.")

        continue


    # --------------------------------------------------------
    # CROP NUMBER PLATE
    # --------------------------------------------------------

    label = read_yolo_label(
        label_path
    )

    plate_crop = crop_plate(
        image,
        label
    )

    if plate_crop.size == 0:

        print("Empty crop.")

        continue


    # --------------------------------------------------------
    # UPSCALE CROP
    # --------------------------------------------------------

    upscaled = cv2.resize(
        plate_crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )


    # --------------------------------------------------------
    # CREATE PREPROCESSING VERSIONS
    # --------------------------------------------------------

    versions = {

        "original":
            upscaled,

        "grayscale":
            grayscale(upscaled),

        "clahe":
            clahe_enhancement(upscaled),

        "threshold":
            threshold_image(upscaled)
    }


    image_result = {

        "image": image_path.name,

        "methods": {}
    }


    # --------------------------------------------------------
    # RUN OCR ON EACH VERSION
    # --------------------------------------------------------

    for method_name, processed_image in versions.items():

        print(
            f"\n  OCR method: "
            f"{method_name}"
        )

        # Save crop for comparison
        crop_filename = (
            f"{image_path.stem}_"
            f"{method_name}.jpg"
        )

        crop_path = (
            CROPS_DIR /
            crop_filename
        )

        cv2.imwrite(
            str(crop_path),
            processed_image
        )


        # OCR
        ocr_results = run_ocr(
            processed_image
        )


        image_result["methods"][
            method_name
        ] = ocr_results


        for result in ocr_results:

            print(
                f"    Text: "
                f"{result['raw_text']} | "
                f"Cleaned: "
                f"{result['cleaned_text']} | "
                f"Confidence: "
                f"{result['confidence']:.4f} | "
                f"Valid plate: "
                f"{result['valid_indian_plate']}"
            )


    all_results.append(
        image_result
    )


# ============================================================
# SAVE JSON REPORT
# ============================================================

json_path = (
    RESULTS_DIR /
    "ocr_comparison.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# TEST TEXT CLEANING FUNCTION
# ============================================================

test_cases = [

    "mh 12 ab 1234",
    "MH-20-CD-5678",
    "mh20cd5678",
    " MH 18 XY 9999 ",
    "MH@12#AB$1234"
]


test_results = []


for sample in test_cases:

    cleaned = clean_plate_text(
        sample
    )

    valid = is_valid_indian_plate(
        cleaned
    )

    test_results.append({

        "input": sample,

        "cleaned": cleaned,

        "valid": valid
    })


tests_path = (
    RESULTS_DIR /
    "text_cleaning_tests.json"
)


with open(
    tests_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        test_results,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 19 OCR COMPARISON COMPLETED")
print("=" * 60)

print(
    f"Images processed: "
    f"{len(all_results)}"
)

print(
    "\nComparison report:"
)

print(json_path)

print(
    "\nText cleaning tests:"
)

print(tests_path)

print(
    "\nCrops saved to:"
)

print(CROPS_DIR)

print("\n" + "=" * 60)