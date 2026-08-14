from ultralytics import YOLO
import csv
import json

# Load YOLO model
model = YOLO("yolo26n.pt")

print("YOLO model loaded successfully.")

# Run detection
results = model.predict(
    source="gray_vehicle.jpg",
    conf=0.25,
    save=True
)

print("Detection completed successfully.")

detection_summary = []

# Extract detection details
for result in results:

    boxes = result.boxes

    if boxes is not None:

        print("Number of detections:", len(boxes))

        for i in range(len(boxes)):

            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].tolist()

            class_name = result.names[cls]

            x1, y1, x2, y2 = xyxy

            print(f"\nDetection {i + 1}")
            print("Class:", class_name)
            print("Class ID:", cls)
            print("Confidence:", round(conf, 4))
            print("Bounding Box:", xyxy)

            detection_summary.append({
                "image": "gray_vehicle.jpg",
                "class": class_name,
                "class_id": cls,
                "confidence": round(conf, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })


# Save CSV
with open(
    "detection_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image",
        "class",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(detection_summary)

print("\nCSV summary saved successfully.")


# Save JSON
with open(
    "detection_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        detection_summary,
        file,
        indent=4
    )

print("JSON summary saved successfully.")

print("\nDay 13 YOLO detection completed successfully.")