from ultralytics import YOLO
import time
import csv

# Load YOLO model
model = YOLO("yolo26n.pt")

print("YOLO model loaded successfully.")

# Video source
video_source = "input/traffic.mp4"

# Start timer
start_time = time.time()

# Run YOLO inference
results = model.predict(
    source=video_source,
    conf=0.25,
    imgsz=640,
    device="cpu",
    stream=True,
    save=True,
    project="runs/detect",
    name="predict",
    exist_ok=True,
    verbose=False
)

frame_number = 0
frame_counts = []

# Process each frame
for result in results:

    frame_number += 1

    vehicle_count = 0

    if result.boxes is not None:

        for i in range(len(result.boxes)):

            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]

            if class_name in [
                "car",
                "motorcycle",
                "bus",
                "truck"
            ]:
                vehicle_count += 1

    frame_counts.append({
        "frame": frame_number,
        "vehicle_count": vehicle_count
    })

    print(
        f"Frame {frame_number}: "
        f"{vehicle_count} vehicles"
    )

# End timer
end_time = time.time()

total_time = end_time - start_time

print("\nVideo processing completed.")

print("Total frames:", frame_number)

print(
    "Total processing time:",
    round(total_time, 2),
    "seconds"
)

if frame_number > 0:

    average_time = total_time / frame_number

    print(
        "Average processing time per frame:",
        round(average_time, 4),
        "seconds"
    )

# Save CSV
csv_file = "frame_vehicle_counts.csv"

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["frame", "vehicle_count"]
    )

    writer.writeheader()
    writer.writerows(frame_counts)

print("Frame count CSV saved successfully.")
print("CSV file:", csv_file)

print("\nDay 14 YOLO video inference completed successfully.")