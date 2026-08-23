# Glove_Detection

# Gloved vs Bare Hand Detection

## 1. Project Overview

This project implements an object detection system for detecting whether workers are wearing gloves.

The system detects two classes:

* `gloved_hand`
* `bare_hand`

The intended application is a safety compliance system that can process images captured from factory cameras or snapshots from video streams.

The complete pipeline includes model training/fine-tuning, image inference, bounding-box detection, annotated image generation, JSON detection logging, and ONNX-based deployment.

---

## 2. Objective

The main objective is to build a computer vision system that can automatically identify gloved and ungloved hands.

For every detected hand, the system provides:

* Class label
* Confidence score
* Bounding box coordinates

Example:

```json
{
    "filename": "image1.jpg",
    "detections": [
        {
            "label": "gloved_hand",
            "confidence": 0.92,
            "bbox": [120, 80, 300, 280]
        },
        {
            "label": "bare_hand",
            "confidence": 0.85,
            "bbox": [400, 100, 570, 320]
        }
    ]
}
```

---

## 3. Dataset

### Dataset Name

**Hand Glove Dataset**

### Source

**Kaggle**

Dataset identifier used in the training notebook:

```text
rsaishivani/hand-glove-dataset
```

The dataset was used for training and evaluation of the hand/glove object detection model.

---

## 4. Model Used

The project uses **YOLO11n** from the Ultralytics framework.

A pretrained YOLO11n model was fine-tuned on the hand-glove dataset.

The training configuration used in the project was:

```text
Model: YOLO11n
Epochs: 5
Image Size: 416 × 416
Batch Size: 16
Device: GPU when available
```

The trained model is saved as:

```text
models/glove_detector.pt
```

---

## 5. Training

The model was initialized using the pretrained YOLO11n model.

Training was performed using the dataset's `data.yaml` configuration.

Example training code:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/content/hand-glove-dataset/data.yaml",
    epochs=5,
    imgsz=416,
    batch=16,
    device=0,
    workers=2,
    project="/content/hand_glove_training",
    name="quick_test",
    patience=3
)
```

After training, the trained model was saved for inference.

---

## 6. Preprocessing

The YOLO preprocessing pipeline is used during training and inference.

Input images are resized to the configured YOLO input size and normalized before being passed to the model.

The model handles image preprocessing internally through the Ultralytics framework.

No manual segmentation of the hands is performed.

---

## 7. Class Handling

The required output classes are:

```text
gloved_hand
bare_hand
```

The project converts dataset/model class names into the required assessment labels.

The label conversion logic handles names such as:

```text
glove
gloved
no glove
bare
```

and maps them to:

```text
glove/gloved → gloved_hand
no glove/bare → bare_hand
```

The exact class order should match the trained model's `data.yaml`/model class definitions.

---

## 8. Project Pipeline

The complete Python inference pipeline is:

```text
Input JPG Images
       ↓
Load YOLO Model
       ↓
Image Preprocessing
       ↓
YOLO Object Detection
       ↓
Confidence Filtering
       ↓
Bounding Box Extraction
       ↓
Class Conversion
       ↓
Draw Bounding Boxes
       ↓
Annotated Images
       +
JSON Detection Logs
```

---

## 9. Project Structure

```text
Part_1_Glove_Detection/
│
├── detection_script.py
├── README.md
│
├── models/
│   └── glove_detector.pt
│
├── input/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
├── output/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── logs/
    ├── image1.json
    ├── image2.json
    └── image3.json
```

---

## 10. Python Requirements

Install the required packages:

```bash
pip install ultralytics opencv-python
```

Python 3.9 or newer is recommended.

---

## 11. Running the Python Detector

Place the trained model inside:

```text
models/glove_detector.pt
```

Place the input images inside:

```text
input/
```

Run the detector using:

```bash
python detection_script.py \
    --input input \
    --output output \
    --confidence 0.25 \
    --model models/glove_detector.pt
```

### Command-line arguments

| Argument       | Description                    |
| -------------- | ------------------------------ |
| `--input`      | Folder containing input images |
| `--output`     | Folder for annotated images    |
| `--confidence` | Detection confidence threshold |
| `--model`      | Path to trained YOLO model     |

Example:

```bash
python detection_script.py --input input --output output --confidence 0.50
```

---

## 12. Annotated Output

The detector saves annotated images inside:

```text
output/
```

Each detected hand is displayed with:

* Bounding box
* Class label
* Confidence score

For example:

```text
gloved_hand: 0.92
```

or:

```text
bare_hand: 0.85
```

---

## 13. JSON Logging

A separate JSON file is generated for every input image.

Example:

```json
{
    "filename": "image1.jpg",
    "detections": [
        {
            "label": "gloved_hand",
            "confidence": 0.92,
            "bbox": [120, 80, 300, 280]
        },
        {
            "label": "bare_hand",
            "confidence": 0.85,
            "bbox": [400, 100, 570, 320]
        }
    ]
}
```

The bounding-box format is:

```text
[x1, y1, x2, y2]
```

where:

* `x1` = left coordinate
* `y1` = top coordinate
* `x2` = right coordinate
* `y2` = bottom coordinate

---

## 14. Model Evaluation

The trained YOLO model was evaluated on the test data using standard object detection metrics.

The evaluation includes:

* Precision
* Recall
* mAP@50
* mAP@50-95

These metrics help measure both classification quality and bounding-box localization performance.

For a safety-compliance application, recall is particularly important because missing an ungloved hand can create a safety risk.

---

## 15. ONNX Conversion

For deployment outside Python, the trained PyTorch model can be exported to ONNX.

Example:

```python
from ultralytics import YOLO

model = YOLO("models/glove_detector.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True
)
```

The generated ONNX model can be stored as:

```text
models/glove_detector.onnx
```

The ONNX model is used by the C++ implementation through OpenCV DNN.

---

## 16. C++ Deployment

The same trained model can be deployed using C++.

The C++ implementation uses:

* C++17
* OpenCV
* OpenCV DNN
* ONNX model
* CMake

The C++ pipeline is:

```text
Input Image
     ↓
OpenCV
     ↓
ONNX YOLO Model
     ↓
Object Detection
     ↓
Confidence Filtering
     ↓
NMS
     ↓
Bounding Boxes
     ↓
Annotated Image + JSON
```

The C++ implementation is provided separately in:

```text
Part_2_Cpp_Glove_Detection/
```

---

## 17. What Worked

The YOLO-based approach provides an end-to-end object detection pipeline with relatively simple training and inference.

The pretrained YOLO11n model provides a useful starting point for fine-tuning on the glove/hand dataset.

The model can generate bounding boxes, class predictions and confidence scores, which directly match the requirements of the assessment.

The trained PyTorch model can also be exported to ONNX, allowing the same model to be used in a C++ deployment environment.

---

## 18. What Did Not Work / Limitations

The training experiment used only five epochs, so the model should be considered a proof-of-concept rather than a production-ready safety system.

Real factory images may differ from the training dataset in lighting, camera angle, background, hand position, motion blur and occlusion.

The model may therefore produce false positives or false negatives when exposed to unseen factory conditions.

More training data and longer training with appropriate hyperparameter tuning would be required before production deployment.

---

## 19. Possible Improvements

Future improvements could include:

1. Collecting factory-specific images.
2. Increasing the number of training epochs.
3. Adding stronger data augmentation.
4. Performing hyperparameter tuning.
5. Adding difficult/edge-case samples.
6. Evaluating the model on a separate real-world factory dataset.
7. Optimizing the ONNX model.
8. Using OpenVINO for CPU deployment.
9. Using GPU inference for high-speed video streams.
10. Adding object tracking for continuous video monitoring.
11. Monitoring false-negative rate for safety compliance.
12. Periodically retraining the model using newly collected factory images.

---

## 20. Safety Considerations

This system is intended as a computer vision assistance system for safety compliance.

A high overall accuracy alone should not be considered sufficient for deployment.

False negatives, particularly undetected bare hands, should be monitored carefully because they may represent a higher operational risk than false positives.

Before production use, the system should be evaluated on representative factory conditions and validated against the required safety standards.

---

## 21. Conclusion

This project demonstrates a complete object detection workflow for gloved and bare hand detection.

The workflow covers dataset preparation, YOLO model fine-tuning, evaluation, image inference, annotated output generation, JSON logging and ONNX conversion.

The same trained model can be deployed using both Python and C++, making the solution suitable for experimentation as well as further deployment development.
