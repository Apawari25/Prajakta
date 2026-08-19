# Day 18 - EasyOCR Fundamentals and First Extraction

## Objective

Use EasyOCR to extract text from number-plate crops and understand:

- Text detection
- Text recognition
- Bounding boxes
- Confidence scores
- Language configuration
- CPU/GPU option
- JSON result storage

## Tools Used

- Python 3.11
- EasyOCR
- OpenCV
- JSON
- YOLO-labelled number-plate dataset

## EasyOCR Configuration

```python
reader = easyocr.Reader(["en"], gpu=False)