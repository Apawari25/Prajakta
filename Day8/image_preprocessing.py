import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = Path("input/gray_vehicle.jpg")
output_dir = Path("output")

output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD IMAGE
# ============================================================

image = cv2.imread(str(input_file))

if image is None:
    raise FileNotFoundError(
        f"Image not found: {input_file}"
    )

print("========== ORIGINAL IMAGE ==========")
print("Image loaded successfully.")
print("Shape:", image.shape)
print("Dimensions:", image.ndim)
print("Data Type:", image.dtype)


# ============================================================
# 3. BGR TO RGB
# ============================================================

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

print("\n========== COLOR FORMAT ==========")
print("OpenCV image format: BGR")
print("Converted image format: RGB")


# ============================================================
# 4. RESIZE
# ============================================================

original_height, original_width = image.shape[:2]

target_width = 800

scale = target_width / original_width

target_height = int(
    original_height * scale
)

resized_image = cv2.resize(
    image,
    (target_width, target_height)
)

cv2.imwrite(
    str(output_dir / "resized.jpg"),
    resized_image
)

print("\n========== RESIZE ==========")
print(
    "Original:",
    original_width,
    "x",
    original_height
)

print(
    "Resized:",
    target_width,
    "x",
    target_height
)


# ============================================================
# 5. GRAYSCALE
# ============================================================

gray_image = cv2.cvtColor(
    resized_image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    str(output_dir / "grayscale.jpg"),
    gray_image
)

print("\n========== GRAYSCALE ==========")
print("Grayscale shape:", gray_image.shape)


# ============================================================
# 6. THRESHOLD
# ============================================================

_, threshold_image = cv2.threshold(
    gray_image,
    120,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    str(output_dir / "threshold.jpg"),
    threshold_image
)

print("\n========== THRESHOLD ==========")
print("Threshold image created.")


# ============================================================
# 7. NUMBER PLATE ROI
# ============================================================

# IMPORTANT:
# The resized image is 800 x 285.
#
# Number plate coordinates are already calculated
# for the resized image.
#
# DO NOT multiply these coordinates by scale.

x1 = 163
y1 = 216

x2 = 338
y2 = 258


# ============================================================
# 8. CHECK RESIZED IMAGE SIZE
# ============================================================

height, width = resized_image.shape[:2]

print("\n========== ROI INFORMATION ==========")

print("Image Width:", width)
print("Image Height:", height)

print("X1:", x1)
print("Y1:", y1)

print("X2:", x2)
print("Y2:", y2)


# ============================================================
# 9. VALIDATE COORDINATES
# ============================================================

if x1 < 0 or y1 < 0:
    raise ValueError(
        "ROI starting coordinates cannot be negative."
    )

if x2 > width or y2 > height:
    raise ValueError(
        f"ROI is outside image boundaries. "
        f"Image = {width}x{height}, "
        f"ROI = ({x1},{y1}) to ({x2},{y2})"
    )

if x2 <= x1:
    raise ValueError(
        "X2 must be greater than X1."
    )

if y2 <= y1:
    raise ValueError(
        "Y2 must be greater than Y1."
    )


# ============================================================
# 10. CROP FULL NUMBER PLATE
# ============================================================

plate_roi = resized_image[
    y1:y2,
    x1:x2
]


# ============================================================
# 11. CHECK CROPPED IMAGE
# ============================================================

print("\n========== PLATE ROI ==========")

print(
    "Cropped ROI Shape:",
    plate_roi.shape
)

print(
    "Cropped ROI Size:",
    plate_roi.size
)

if plate_roi.size == 0:
    raise ValueError(
        "ROI crop is empty."
    )


# ============================================================
# 12. SAVE NUMBER PLATE
# ============================================================

cv2.imwrite(
    str(output_dir / "plate_roi.jpg"),
    plate_roi
)

print(
    "Number plate cropped successfully!"
)


# ============================================================
# 13. GRAYSCALE PLATE
# ============================================================

plate_gray = cv2.cvtColor(
    plate_roi,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    str(output_dir / "plate_gray.jpg"),
    plate_gray
)

print("\n========== PLATE GRAYSCALE ==========")

print(
    "Plate grayscale image created."
)


# ============================================================
# 14. THRESHOLD PLATE
# ============================================================

_, plate_threshold = cv2.threshold(
    plate_gray,
    120,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    str(output_dir / "plate_threshold.jpg"),
    plate_threshold
)

print("\n========== PLATE THRESHOLD ==========")

print(
    "Plate threshold image created."
)


# ============================================================
# 15. SAVE RGB IMAGE
# ============================================================

rgb_for_save = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    str(output_dir / "rgb_image.jpg"),
    rgb_for_save
)

print("\n========== RGB IMAGE ==========")

print(
    "RGB image saved successfully."
)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n========== PROCESSING COMPLETE ==========")

print("Output files:")

for file in sorted(output_dir.iterdir()):
    print("-", file.name)

print(
    "\nAll image preprocessing steps completed successfully!"
)