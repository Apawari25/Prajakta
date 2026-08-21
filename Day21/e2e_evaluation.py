import json
import re
from pathlib import Path
from datetime import datetime


# ============================================================
# DAY 21 - END-TO-END ANPR EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DAY20_JSON = BASE_DIR.parent / "Day20" / "results" / "anpr_results.json"
OCR_REPORT = BASE_DIR / "results" / "ocr_evaluation.md"

OUTPUT_JSON = BASE_DIR / "results" / "e2e_evaluation.json"
OUTPUT_MD = BASE_DIR / "results" / "e2e_evaluation.md"


# ------------------------------------------------------------
# Normalize plate text
# ------------------------------------------------------------

def normalize_plate(text):
    if not text:
        return ""

    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ------------------------------------------------------------
# Read OCR ground truth from Day21 OCR evaluation report
# ------------------------------------------------------------

def load_ground_truth():
    ground_truth = {}

    if not OCR_REPORT.exists():
        print("WARNING: OCR evaluation report not found.")
        return ground_truth

    content = OCR_REPORT.read_text(encoding="utf-8")

    for line in content.splitlines():

        if not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.split("|")]

        # Expected:
        # | car1.jpg | MH11 | MH1I | 0.4029 | False |

        if len(parts) < 6:
            continue

        image = parts[1]

        if not image.lower().endswith(".jpg"):
            continue

        gt = parts[2]

        if not gt or gt.lower() in ["ground truth", "---"]:
            continue

        ground_truth[image] = normalize_plate(gt)

    return ground_truth


# ------------------------------------------------------------
# Load Day20 ANPR results
# ------------------------------------------------------------

def load_anpr_results():

    if not DAY20_JSON.exists():
        raise FileNotFoundError(
            f"Day20 result file not found:\n{DAY20_JSON}"
        )

    with open(DAY20_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Group detections by image
# ------------------------------------------------------------

def group_by_image(results):

    grouped = {}

    for item in results:

        detection = item.get("detection_result", {})

        image = detection.get("image")

        if not image:
            continue

        grouped.setdefault(image, []).append(detection)

    return grouped


# ------------------------------------------------------------
# Select best detection
# ------------------------------------------------------------

def select_best_detection(detections):

    if not detections:
        return None

    # Prefer highest OCR confidence
    return max(
        detections,
        key=lambda x: float(x.get("ocr_confidence", 0))
    )


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("DAY 21 - END-TO-END ANPR EVALUATION")
    print("=" * 70)

    # Load data
    results = load_anpr_results()
    ground_truth = load_ground_truth()

    grouped = group_by_image(results)

    print(f"\nANPR result records : {len(results)}")
    print(f"Unique images       : {len(grouped)}")
    print(f"Ground truth images  : {len(ground_truth)}")

    detailed_results = []

    detected_images = 0
    end_to_end_correct = 0
    total_images = len(ground_truth)

    confidence_values = []

    # --------------------------------------------------------
    # Evaluate each ground-truth image
    # --------------------------------------------------------

    for image, gt in sorted(ground_truth.items()):

        detections = grouped.get(image, [])

        if detections:
            detected_images += 1

        best = select_best_detection(detections)

        if best:

            prediction = normalize_plate(
                best.get("ocr_text", "")
            )

            yolo_conf = float(
                best.get("yolo_confidence", 0)
            )

            ocr_conf = float(
                best.get("ocr_confidence", 0)
            )

            confidence_values.append(ocr_conf)

            exact_match = prediction == gt

            if exact_match:
                end_to_end_correct += 1

            detailed_results.append({
                "image": image,
                "ground_truth": gt,
                "prediction": prediction,
                "yolo_confidence": round(yolo_conf, 4),
                "ocr_confidence": round(ocr_conf, 4),
                "detected": True,
                "exact_match": exact_match,
                "valid_indian_plate": best.get(
                    "valid_indian_plate",
                    False
                )
            })

        else:

            detailed_results.append({
                "image": image,
                "ground_truth": gt,
                "prediction": "",
                "yolo_confidence": 0,
                "ocr_confidence": 0,
                "detected": False,
                "exact_match": False,
                "valid_indian_plate": False
            })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if total_images > 0:
        detection_recall = (
            detected_images / total_images
        ) * 100

        end_to_end_accuracy = (
            end_to_end_correct / total_images
        ) * 100

    else:
        detection_recall = 0
        end_to_end_accuracy = 0

    if confidence_values:
        average_ocr_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )
    else:
        average_ocr_confidence = 0

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "total_images": total_images,
        "detected_images": detected_images,
        "detection_recall_percent": round(
            detection_recall, 2
        ),
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy_percent": round(
            end_to_end_accuracy, 2
        ),
        "average_ocr_confidence": round(
            average_ocr_confidence, 4
        ),
        "detailed_results": detailed_results
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save Markdown report
    # --------------------------------------------------------

    md = []

    md.append("# Day 21 - End-to-End ANPR Evaluation\n")

    md.append("## Summary\n")

    md.append(
        f"- Total images: **{total_images}**"
    )

    md.append(
        f"- Detected images: **{detected_images}**"
    )

    md.append(
        f"- Detection recall: **{detection_recall:.2f}%**"
    )

    md.append(
        f"- End-to-end correct: **{end_to_end_correct}**"
    )

    md.append(
        f"- End-to-end accuracy: "
        f"**{end_to_end_accuracy:.2f}%**"
    )

    md.append(
        f"- Average OCR confidence: "
        f"**{average_ocr_confidence:.4f}**\n"
    )

    md.append("## Detailed Results\n")

    md.append(
        "| Image | Ground Truth | Prediction | "
        "YOLO Conf. | OCR Conf. | Detected | Exact Match |"
    )

    md.append(
        "|---|---|---|---:|---:|---|---|"
    )

    for item in detailed_results:

        md.append(
            f"| {item['image']} | "
            f"{item['ground_truth']} | "
            f"{item['prediction']} | "
            f"{item['yolo_confidence']:.4f} | "
            f"{item['ocr_confidence']:.4f} | "
            f"{item['detected']} | "
            f"{item['exact_match']} |"
        )

    OUTPUT_MD.write_text(
        "\n".join(md),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL END-TO-END RESULTS")
    print("=" * 70)

    print(f"Total images          : {total_images}")
    print(f"Detected images       : {detected_images}")
    print(
        f"Detection recall      : "
        f"{detection_recall:.2f}%"
    )

    print(
        f"End-to-end correct    : "
        f"{end_to_end_correct}"
    )

    print(
        f"End-to-end accuracy   : "
        f"{end_to_end_accuracy:.2f}%"
    )

    print(
        f"Average OCR confidence: "
        f"{average_ocr_confidence:.4f}"
    )

    print("\nJSON report:")
    print(OUTPUT_JSON)

    print("\nMarkdown report:")
    print(OUTPUT_MD)

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()