import cv2
import easyocr
import json
import re
import logging
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# DAY 20 - YOLO + EASYOCR ANPR PIPELINE
# ============================================================

print("=" * 70)
print("DAY 20 - END-TO-END ANPR PIPELINE")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\OCR_YOLO_Training")

DAY16_DIR = PROJECT_ROOT / "Day16"
DAY20_DIR = PROJECT_ROOT / "Day20"

MODEL_PATH = (
    DAY16_DIR
    / "runs"
    / "detect"
    / "runs"
    / "number_plate_detector"
    / "weights"
    / "best.pt"
)

INPUT_DIR = DAY16_DIR / "dataset" / "images" / "val"

RESULTS_DIR = DAY20_DIR / "results"
OUTPUT_DIR = DAY20_DIR / "output"
LOG_DIR = DAY20_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "anpr_results.json"
LOG_FILE = LOG_DIR / "anpr_pipeline.log"


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("ANPR pipeline started.")


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input directory not found:\n{INPUT_DIR}"
    )


print("\nYOLO model:")
print(MODEL_PATH)

print("\nInput directory:")
print(INPUT_DIR)


# ============================================================
# 4. LOAD YOLO MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")

logging.info("YOLO model loaded successfully.")


# ============================================================
# 5. LOAD EASYOCR
# ============================================================

print("\nLoading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded successfully.")

logging.info("EasyOCR loaded successfully.")


# ============================================================
# 6. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize OCR output.
    """

    if text is None:
        return ""

    text = str(text).upper()

    # Remove spaces and special characters
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# 7. INDIAN PLATE VALIDATION
# ============================================================

def validate_indian_plate(text):
    """
    Basic Indian vehicle registration pattern.

    Example:
    MH12AB1234
    """

    text = normalize_text(text)

    pattern = (
        r"^[A-Z]{2}"
        r"[0-9]{1,2}"
        r"[A-Z]{1,3}"
        r"[0-9]{1,4}$"
    )

    return bool(
        re.fullmatch(
            pattern,
            text
        )
    )


# ============================================================
# 8. SAFE CROP FUNCTION
# ============================================================

def safe_crop(image, x1, y1, x2, y2):

    height, width = image.shape[:2]

    # Clamp coordinates to image boundaries
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))

    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    # Invalid box
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# 9. PREPROCESS OCR IMAGE
# ============================================================

def preprocess_for_ocr(crop):

    # Upscale
    resized = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# ============================================================
# 10. OCR FUNCTION
# ============================================================

def run_ocr(crop):

    processed = preprocess_for_ocr(
        crop
    )

    ocr_results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    best_text = ""
    best_confidence = 0.0
    best_bbox = []

    for detection in ocr_results:

        bbox, text, confidence = detection

        confidence = float(
            confidence
        )

        normalized = normalize_text(
            text
        )

        if confidence > best_confidence:

            best_text = normalized
            best_confidence = confidence

            best_bbox = [
                [
                    int(point[0]),
                    int(point[1])
                ]
                for point in bbox
            ]

    return (
        best_text,
        best_confidence,
        best_bbox
    )


# ============================================================
# 11. FIND INPUT IMAGES
# ============================================================

image_paths = sorted(
    INPUT_DIR.glob("*.jpg")
)

print(
    f"\nTotal input images: "
    f"{len(image_paths)}"
)

logging.info(
    f"Found {len(image_paths)} input images."
)


# ============================================================
# 12. PROCESS IMAGES
# ============================================================

all_results = []


for image_number, image_path in enumerate(
    image_paths,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"Processing "
        f"{image_number}/{len(image_paths)}: "
        f"{image_path.name}"
    )

    logging.info(
        f"Processing {image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("ERROR: Could not read image.")

        logging.error(
            f"Could not read {image_path.name}"
        )

        continue


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    try:

        prediction = model.predict(
            source=image,
            conf=0.05,
            iou=0.7,
            verbose=False
        )

    except Exception as error:

        logging.exception(
            f"YOLO error for {image_path.name}: {error}"
        )

        continue


    image_results = []


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in prediction:

        if result.boxes is None:
            continue

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = coordinates

            yolo_confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )


            # ------------------------------------------------
            # SAFE CROP
            # ------------------------------------------------

            crop = safe_crop(
                image,
                x1,
                y1,
                x2,
                y2
            )

            if crop is None:

                logging.warning(
                    f"Invalid crop for {image_path.name}"
                )

                continue


            # ------------------------------------------------
            # SAVE CROPPED PLATE
            # ------------------------------------------------

            crop_filename = (
                f"{image_path.stem}_"
                f"plate_{len(image_results) + 1}.jpg"
            )

            crop_path = (
                OUTPUT_DIR /
                crop_filename
            )

            cv2.imwrite(
                str(crop_path),
                crop
            )


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            (
                detected_text,
                ocr_confidence,
                ocr_bbox
            ) = run_ocr(
                crop
            )


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            valid_plate = (
                validate_indian_plate(
                    detected_text
                )
            )


            # ------------------------------------------------
            # ANNOTATE IMAGE
            # ------------------------------------------------

            start_point = (
                int(x1),
                int(y1)
            )

            end_point = (
                int(x2),
                int(y2)
            )

            cv2.rectangle(
                image,
                start_point,
                end_point,
                (0, 255, 0),
                2
            )


            label = (
                f"{detected_text} "
                f"| OCR: {ocr_confidence:.2f}"
            )

            cv2.putText(
                image,
                label,
                (
                    int(x1),
                    max(
                        20,
                        int(y1) - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # RESULT OBJECT
            # ------------------------------------------------

            detection_result = {

                "detection_result": {
                    "image": image_path.name,

                    "timestamp":
                        datetime.now().isoformat(),

                    "bounding_box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },

                    "yolo_confidence":
                        yolo_confidence,

                    "ocr_text":
                        detected_text,

                    "ocr_confidence":
                        ocr_confidence,

                    "ocr_bounding_box":
                        ocr_bbox,

                    "valid_indian_plate":
                        valid_plate
                }
            }


            image_results.append(
                detection_result
            )


            print(
                f"YOLO confidence: "
                f"{yolo_confidence:.4f}"
            )

            print(
                f"OCR text: "
                f"{detected_text}"
            )

            print(
                f"OCR confidence: "
                f"{ocr_confidence:.4f}"
            )

            print(
                f"Valid plate: "
                f"{valid_plate}"
            )


    # ========================================================
    # SAVE ANNOTATED IMAGE
    # ========================================================

    annotated_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_annotated.jpg"
    )

    cv2.imwrite(
        str(annotated_path),
        image
    )


    # ========================================================
    # ADD IMAGE RESULTS
    # ========================================================

    all_results.extend(
        image_results
    )


# ============================================================
# 13. SAVE JSON
# ============================================================

with open(
    JSON_OUTPUT,
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
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 20 ANPR PIPELINE COMPLETED")
print("=" * 70)

print(
    f"Images processed: "
    f"{len(image_paths)}"
)

print(
    f"Detection results: "
    f"{len(all_results)}"
)

print(
    "\nJSON results:"
)

print(JSON_OUTPUT)

print(
    "\nAnnotated output:"
)

print(OUTPUT_DIR)

print(
    "\nLog file:"
)

print(LOG_FILE)

print("=" * 70)

logging.info(
    "ANPR pipeline completed successfully."
)