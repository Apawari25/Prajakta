from ultralytics import YOLO
from pathlib import Path
import shutil

# ============================================
# DAY 17 - VALIDATE, IMPROVE AND EXPORT
# ============================================

# Project paths
BASE_DIR = Path(r"C:\OCR_YOLO_Training")

MODEL_PATH = (
    BASE_DIR
    / "Day16"
    / "runs"
    / "detect"
    / "runs"
    / "number_plate_detector"
    / "weights"
    / "best.pt"
)

DATA_YAML = BASE_DIR / "Day16" / "data.yaml"

DAY17_DIR = BASE_DIR / "Day17"

EVALUATION_DIR = DAY17_DIR / "evaluation"
PREDICTIONS_DIR = DAY17_DIR / "predictions"
EXPORTS_DIR = DAY17_DIR / "exports"

# Create folders
EVALUATION_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("DAY 17 - YOLO VALIDATION AND EXPORT")
print("=" * 60)

# ============================================
# 1. CHECK MODEL
# ============================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.yaml not found:\n{DATA_YAML}"
    )

print("\nModel found:")
print(MODEL_PATH)

print("\nDataset configuration:")
print(DATA_YAML)

# ============================================
# 2. LOAD BEST MODEL
# ============================================

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

# ============================================
# 3. VALIDATE MODEL
# ============================================

print("\n" + "=" * 60)
print("VALIDATION STARTED")
print("=" * 60)

metrics = model.val(
    data=str(DATA_YAML),
    imgsz=640,
    conf=0.25,
    iou=0.7,
    plots=True,
    project=str(EVALUATION_DIR),
    name="validation",
    exist_ok=True
)

# ============================================
# 4. DISPLAY METRICS
# ============================================

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"mAP50         : {map50:.4f}")
print(f"mAP50-95      : {map50_95:.4f}")

# ============================================
# 5. SAVE EVALUATION REPORT
# ============================================

report_path = EVALUATION_DIR / "evaluation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("DAY 17 - YOLO MODEL EVALUATION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Model: {MODEL_PATH}\n")
    file.write(f"Dataset: {DATA_YAML}\n\n")

    file.write("Evaluation Metrics\n")
    file.write("-" * 30 + "\n")

    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"mAP50     : {map50:.4f}\n")
    file.write(f"mAP50-95  : {map50_95:.4f}\n")

print("\nEvaluation report saved:")
print(report_path)

# ============================================
# 6. PREDICTION ON VALIDATION IMAGES
# ============================================

VAL_IMAGES = BASE_DIR / "Day16" / "dataset" / "images" / "val"

print("\n" + "=" * 60)
print("RUNNING PREDICTIONS")
print("=" * 60)

model.predict(
    source=str(VAL_IMAGES),
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(PREDICTIONS_DIR),
    name="val_predictions",
    exist_ok=True
)

print("\nPrediction completed.")

# ============================================
# 7. EXPORT MODEL TO ONNX
# ============================================

print("\n" + "=" * 60)
print("EXPORTING MODEL TO ONNX")
print("=" * 60)

onnx_path = model.export(
    format="onnx",
    imgsz=640,
    opset=12
)

print("\nONNX export completed.")
print("Original ONNX path:")
print(onnx_path)

# ============================================
# 8. COPY ONNX FILE TO DAY17/exports
# ============================================

onnx_path = Path(onnx_path)

if onnx_path.exists():

    destination = EXPORTS_DIR / onnx_path.name

    shutil.copy2(
        onnx_path,
        destination
    )

    print("\nONNX model copied to:")
    print(destination)

else:

    print("\nWARNING: ONNX file was not found.")

# ============================================
# 9. FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("DAY 17 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nEvaluation report:")
print(report_path)

print("\nPredictions:")
print(PREDICTIONS_DIR)

print("\nONNX export:")
print(EXPORTS_DIR)

print("\nMetrics:")
print(f"Precision    = {precision:.4f}")
print(f"Recall       = {recall:.4f}")
print(f"mAP50        = {map50:.4f}")
print(f"mAP50-95     = {map50_95:.4f}")

print("\nNext step: Review prediction images and failure cases.")