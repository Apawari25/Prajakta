# ANPR - YOLO + EasyOCR

## Project Overview

This project implements an Automatic Number Plate Recognition (ANPR)
pipeline using YOLO for number plate detection and EasyOCR for text
recognition.

The pipeline performs:

1. Number plate detection using YOLO
2. Safe bounding-box cropping
3. Image preprocessing using OpenCV
4. OCR using EasyOCR
5. Text normalization
6. Indian number plate validation
7. Confidence filtering
8. Structured JSON results
9. Evaluation of detection and OCR performance

---

## Project Structure

```text
Day21/
│
├── config/
├── demo/
│   ├── input/
│   └── output/
│
├── failure_gallery/
├── metrics/
├── models/
├── results/
├── tests/
│
├── evaluation.py
├── ocr_evaluation.py
├── e2e_evaluation.py
├── requirements.txt
└── README.md