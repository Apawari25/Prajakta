from ultralytics import YOLO


# ==========================================
# DAY 16 - YOLO NUMBER PLATE DETECTOR
# ==========================================

# 1. Load pretrained YOLO model
model = YOLO("yolo26n.pt")


# 2. Train the model
results = model.train(
    data="data.yaml",
    epochs=50,
    batch=4,
    imgsz=640,
    patience=10,
    project="runs",
    name="number_plate_detector",
    exist_ok=True
)


# 3. Training completed
print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")
print("Best model saved at:")
print("runs/number_plate_detector/weights/best.pt")