import json
import time
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# DAY 21 - FINAL MODEL EVALUATION
# ============================================================

PROJECT_ROOT = Path(r"C:\OCR_YOLO_Training")

DAY16_DIR = PROJECT_ROOT / "Day16"
DAY21_DIR = PROJECT_ROOT / "Day21"

MODEL_PATH = (
    DAY16_DIR
    / "runs"
    / "detect"
    / "runs"
    / "number_plate_detector"
    / "weights"
    / "best.pt"
)

DATA_YAML = DAY16_DIR / "data.yaml"

METRICS_DIR = DAY21_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = METRICS_DIR / "evaluation_report.json"
REPORT_MD = METRICS_DIR / "evaluation_report.md"


print("=" * 70)
print("DAY 21 - FINAL YOLO EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset configuration not found:\n{DATA_YAML}"
    )


print("\nModel:")
print(MODEL_PATH)

print("\nDataset:")
print(DATA_YAML)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

start_time = time.perf_counter()

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    conf=0.25,
    verbose=True
)

end_time = time.perf_counter()

total_validation_time = (
    end_time - start_time
)


# ============================================================
# EXTRACT METRICS
# ============================================================

precision = float(
    metrics.box.mp
)

recall = float(
    metrics.box.mr
)

map50 = float(
    metrics.box.map50
)

map50_95 = float(
    metrics.box.map
)


# ============================================================
# SPEED
# ============================================================

speed = getattr(
    metrics,
    "speed",
    {}
)

preprocess_time = float(
    speed.get("preprocess", 0)
)

inference_time = float(
    speed.get("inference", 0)
)

postprocess_time = float(
    speed.get("postprocess", 0)
)

total_ms_per_image = (
    preprocess_time
    + inference_time
    + postprocess_time
)

if total_ms_per_image > 0:

    fps = (
        1000 /
        total_ms_per_image
    )

else:

    fps = 0


# ============================================================
# RESULT DICTIONARY
# ============================================================

report = {

    "project": "Number Plate ANPR",

    "model": str(MODEL_PATH),

    "dataset": str(DATA_YAML),

    "evaluation": {

        "precision": precision,

        "recall": recall,

        "mAP50": map50,

        "mAP50_95": map50_95
    },

    "speed": {

        "preprocess_ms": preprocess_time,

        "inference_ms": inference_time,

        "postprocess_ms": postprocess_time,

        "total_ms_per_image":
            total_ms_per_image,

        "estimated_fps": fps
    },

    "validation_runtime_seconds":
        total_validation_time,

    "note":
        "Metrics are reported from the available validation dataset. "
        "No fabricated accuracy values are used."
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# SAVE MARKDOWN REPORT
# ============================================================

markdown_report = f"""
# Day 21 - Final Evaluation Report

## Project

Number Plate ANPR using YOLO and EasyOCR.

## Model

`{MODEL_PATH}`

## Dataset

`{DATA_YAML}`

## YOLO Detection Metrics

| Metric | Result |
|---|---:|
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| mAP50 | {map50:.4f} |
| mAP50-95 | {map50_95:.4f} |

## Processing Speed

| Metric | Result |
|---|---:|
| Preprocess | {preprocess_time:.2f} ms |
| Inference | {inference_time:.2f} ms |
| Postprocess | {postprocess_time:.2f} ms |
| Total | {total_ms_per_image:.2f} ms/image |
| Estimated FPS | {fps:.2f} |

## Evaluation Runtime

{total_validation_time:.2f} seconds

## Limitations

The current validation dataset is small and does not represent
all real-world conditions such as night scenes, extreme angles,
blur, occlusion and different plate formats.

OCR exact-match and end-to-end accuracy require a labelled
evaluation set containing ground-truth plate text.

## Next Improvements

1. Increase the number of labelled training images.
2. Add day, night and different-angle samples.
3. Improve YOLO plate detection.
4. Tune confidence and IoU thresholds.
5. Improve OCR preprocessing.
6. Add a larger ground-truth OCR test set.
7. Measure end-to-end ANPR accuracy.
8. Benchmark ONNX inference speed.
"""


with open(
    REPORT_MD,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        markdown_report.strip()
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"mAP50     : {map50:.4f}"
)

print(
    f"mAP50-95  : {map50_95:.4f}"
)

print(
    f"\nSpeed     : {total_ms_per_image:.2f} ms/image"
)

print(
    f"FPS       : {fps:.2f}"
)

print(
    "\nJSON report:"
)

print(REPORT_JSON)

print(
    "\nMarkdown report:"
)

print(REPORT_MD)

print("\n" + "=" * 70)
print("DAY 21 EVALUATION COMPLETED")
print("=" * 70)