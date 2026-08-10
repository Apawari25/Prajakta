import cv2
import numpy as np
print("OpenCV version:", cv2.__version__)

# Load image
image = cv2.imread("Skoda-Superb-BW-India.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully.")
    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Dimensions:", image.ndim)

    # Resize image

resized = cv2.resize(image, (700, 250))

print("Resized shape:", resized.shape)

cv2.imwrite("resized_vehicle.jpg", resized)

print("Resized image saved successfully.")

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("RGB image shape:", rgb_image.shape)

cv2.imwrite("rgb_vehicle.jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


# Convert BGR to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Grayscale shape:", gray_image.shape)

cv2.imwrite("gray_vehicle.jpg", gray_image)

print("RGB and grayscale images saved successfully.")

# Thresholding

_, threshold_image = cv2.threshold(
    gray_image,
    127,
    255,
    cv2.THRESH_BINARY
)

print("Threshold image shape:", threshold_image.shape)

cv2.imwrite("threshold_vehicle.jpg", threshold_image)

print("Threshold image saved successfully.")

# Crop Region of Interest (ROI)

roi = image[300:450, 500:1200]

print("ROI shape:", roi.shape)

cv2.imwrite("plate_roi.jpg", roi)

print("ROI saved successfully.")

# Safe ROI coordinates using NumPy clip

height, width = image.shape[:2]

x1 = int(np.clip(500, 0, width))
x2 = int(np.clip(1200, 0, width))

y1 = int(np.clip(300, 0, height))
y2 = int(np.clip(450, 0, height))

safe_roi = image[y1:y2, x1:x2]

print("Safe ROI shape:", safe_roi.shape)

cv2.imwrite("safe_plate_roi.jpg", safe_roi)

print("Safe ROI saved successfully.")