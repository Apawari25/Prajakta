import json
import re
from pathlib import Path


# ============================================================
# DAY 21 - OCR EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_FILE = BASE_DIR.parent / "Day18" / "results" / "ocr_results.json"
RESULTS_DIR = BASE_DIR / "results"
REPORT_FILE = RESULTS_DIR / "ocr_evaluation.md"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GROUND TRUTH
# ============================================================

GROUND_TRUTH = {
    "car1.jpg": "MH11",
    "car2.jpg": "MH12",
    "car3.jpg": "MH13",
    "car4.jpg": "MH14",
    "car5.jpg": "MH15",
    "car6.jpg": "MH16",
    "car7.jpg": "MH17",
    "car8.jpg": "MH18",
    "car9.jpg": "MH19",
    "car10.jpg": "MH20",
    "car11.jpg": "MH21",
    "car12.jpg": "MH22",
    "car13.jpg": "MH23",
    "car14.jpg": "MH24",
    "car15.jpg": "MH25",
    "car16.jpg": "MH26",
    "car17.jpg": "MH27",
    "car18.jpg": "MH28",
    "car19.jpg": "MH29",
    "car20.jpg": "MH30",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize OCR text before comparison.

    Example:
        "MH 20" -> "MH20"
        "mh20"  -> "MH20"
    """

    if not text:
        return ""

    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# LOAD OCR RESULTS
# ============================================================

print("=" * 70)
print("DAY 21 - OCR EVALUATION")
print("=" * 70)

if not OCR_FILE.exists():
    raise FileNotFoundError(
        f"OCR results not found:\n{OCR_FILE}"
    )

with open(OCR_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# PROCESS RESULTS
# ============================================================

evaluation_results = []

for item in data:

    image_name = item.get("image", "")

    ocr_results = item.get("ocr_results", [])

    prediction = ""
    confidence = 0.0

    if ocr_results:

        # Select OCR result with highest confidence
        best_result = max(
            ocr_results,
            key=lambda x: x.get("confidence", 0.0)
        )

        prediction = best_result.get("text", "")
        confidence = float(
            best_result.get("confidence", 0.0)
        )

    ground_truth = GROUND_TRUTH.get(image_name, "")

    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truth)

    exact_match = (
        normalized_prediction == normalized_ground_truth
        and normalized_ground_truth != ""
    )

    evaluation_results.append({
        "image": image_name,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "confidence": confidence,
        "exact_match": exact_match
    })


# ============================================================
# METRICS
# ============================================================

total_images = len(evaluation_results)

exact_matches = sum(
    1
    for result in evaluation_results
    if result["exact_match"]
)

exact_match_rate = (
    exact_matches / total_images * 100
    if total_images > 0
    else 0
)

average_confidence = (
    sum(
        result["confidence"]
        for result in evaluation_results
    ) / total_images
    if total_images > 0
    else 0
)


# ============================================================
# SAVE MARKDOWN REPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("# Day 21 - OCR Evaluation\n\n")

    f.write("## Summary\n\n")

    f.write(f"- Total images: **{total_images}**\n")
    f.write(f"- Exact matches: **{exact_matches}**\n")
    f.write(
        f"- Exact match rate: **{exact_match_rate:.2f}%**\n"
    )
    f.write(
        f"- Average OCR confidence: **{average_confidence:.4f}**\n\n"
    )

    f.write("## Detailed Results\n\n")

    f.write(
        "| Image | Ground Truth | Prediction | Confidence | Exact Match |\n"
    )

    f.write(
        "|---|---|---|---:|---|\n"
    )

    for result in evaluation_results:

        f.write(
            f"| {result['image']} "
            f"| {result['ground_truth']} "
            f"| {result['prediction']} "
            f"| {result['confidence']:.4f} "
            f"| {result['exact_match']} |\n"
        )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("OCR EVALUATION COMPLETED")
print("=" * 70)

print(f"Total images       : {total_images}")
print(f"Exact matches      : {exact_matches}")
print(f"Exact match rate   : {exact_match_rate:.2f}%")
print(f"Average confidence : {average_confidence:.4f}")

print()
print("Report saved to:")
print(REPORT_FILE)

print("=" * 70)