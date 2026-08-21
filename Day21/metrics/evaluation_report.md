# Day 21 - Final Evaluation Report

## Project

Number Plate ANPR using YOLO and EasyOCR.

## Model

`C:\OCR_YOLO_Training\Day16\runs\detect\runs\number_plate_detector\weights\best.pt`

## Dataset

`C:\OCR_YOLO_Training\Day16\data.yaml`

## YOLO Detection Metrics

| Metric | Result |
|---|---:|
| Precision | 0.0000 |
| Recall | 0.0000 |
| mAP50 | 0.0000 |
| mAP50-95 | 0.0000 |

## Processing Speed

| Metric | Result |
|---|---:|
| Preprocess | 6.88 ms |
| Inference | 385.29 ms |
| Postprocess | 2.74 ms |
| Total | 394.91 ms/image |
| Estimated FPS | 2.53 |

## Evaluation Runtime

13.90 seconds

## Limitations

The current validation dataset is small and does not represent
all real-world conditions such as night scenes, extreme angles,
blur, occlusion and different plate formats.

OCR exact-match and end-to-end accuracy require a labelled
evaluation set containing ground-truth plate text.

## Next Improvements

1. Increase the number of labelled training images.
2. Add day, night and different-angle samples.
3. Improve YOLO plate detection.
4. Tune confidence and IoU thresholds.
5. Improve OCR preprocessing.
6. Add a larger ground-truth OCR test set.
7. Measure end-to-end ANPR accuracy.
8. Benchmark ONNX inference speed.