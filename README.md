# 🐝 Bee Monitoring Pipeline for Environmental Observation

A deep learning-based computer vision pipeline for automatic pollinator detection and monitoring from field videos. This project aims to support biodiversity research and environmental observation by detecting and classifying pollinators using state-of-the-art object detection models.

---

## 📌 Project Overview

Monitoring pollinators such as bees is essential for understanding ecosystem health and biodiversity. Manual observation is time-consuming and difficult to scale. This project develops an automated monitoring pipeline capable of detecting pollinators from field recordings using modern deep learning techniques.

The project was developed as part of the **MORO Project** at the **University of Bonn**.

---

## 🎯 Objectives

- Build an end-to-end bee monitoring pipeline
- Train and compare multiple object detection models
- Improve detection accuracy using Hard Negative Mining
- Evaluate model performance using standard detection metrics
- Prepare the pipeline for future deployment on edge devices

---

## 🏗️ Pipeline

```
Field Videos
      │
      ▼
Frame Extraction
      │
      ▼
Dataset Annotation
      │
      ▼
Model Training
      │
      ▼
Object Detection
      │
      ▼
Performance Evaluation
      │
      ▼
Monitoring & Analysis
```

---

## 📂 Dataset

The dataset consists of field recordings containing pollinating insects collected under real-world environmental conditions.

### Classes

- Bee
- Bumblebee
- Hoverfly
- Other Insect
- Moth

The dataset was manually annotated in YOLO format and further improved using Hard Negative Mining to reduce false positive detections.

---
## 🤖 Models

The trained model weights are not included in this repository because they exceed GitHub's file size limit.

You can download the pretrained weights from Google Drive.

| Model | Download |
|-------|----------|
| YOLO11x | https://drive.google.com/file/d/1xV4-iaVbs1vrGKcq5Fk-BA2mhyGmiUT8/view?usp=sharing |
| YOLO26x | https://drive.google.com/file/d/1guemxRUqqwk9p6cDBaxCtWghELpJboAN/view?usp=sharing |
| RF-DETR | https://drive.google.com/file/d/1wIGSBCD4cjMfEY9hkn5lLV-aDT988RE1/view?usp=sharing |

After downloading, place the weights inside the `models/` directory.

```
models/
├── yolov11x.pt
├── yolov26x.pt
└── rfdetr_best.pth
```

---

### Running Inference

Example (run from the project root):

```bash
python src/hard_negative_mining.py infer \
  --weights models/yolo26x.pt \
  --source videos \
  --output hard_negative_output
```

### Hard-negative review and dataset preparation

The inference command accepts one supported video or a folder of videos. It samples
frames at a configurable interval, skips near-identical sampled frames, and saves
only frames with detections. Adjust the confidence, IoU, image-size, and sampling
settings as needed:

```bash
python src/hard_negative_mining.py infer \
  --weights models/yolo26x.pt \
  --source videos \
  --output hard_negative_output \
  --conf 0.25 --iou 0.7 --imgsz 960 --frame-interval 2
```

Inference creates `review.csv` (one row per candidate frame), `predictions.csv`
(class, confidence, and pixel bounding boxes), plus `frames/`, `annotated/`, and
`labels/` folders grouped by video and frame number. Candidates start as
`unreviewed`; predictions are never automatically marked as false positives.
Review each annotated frame and its label file, then set `review_status` in
`review.csv` to `true_positive`, `false_positive`, or `uncertain`. For true
positives, correct the YOLO label file to include **all** target insects and their
correct class IDs and boxes. Mark a frame `false_positive` only after verifying
that it contains no target insects, and clear its label file so it is empty.
Unreviewed and uncertain frames are excluded from the dataset.

Create a separate reviewed-frame dataset using the class names and IDs from the
existing YOLO configuration (`config/YOLOruns.yaml`):

```bash
python src/hard_negative_mining.py prepare \
  --review-csv hard_negative_output/review.csv \
  --data config/YOLOruns.yaml \
  --output hard_negative_dataset
```

The generated dataset contains `images/{train,val,test}`, matching
`labels/{train,val,test}`, and a ready-to-use `data.yaml`. At least three reviewed
videos are required; all frames from a video are assigned to the same split.
The output is an additive dataset of reviewed frames; the original dataset and
its annotations are not copied or modified. Outputs are not overwritten, so use
a new output directory for each run. Training is not started automatically.

---

### Notes

- The Google Drive links contain the final trained models.
- The repository contains all training and inference scripts required to reproduce the results.
- If you train the models yourself, simply replace the downloaded weights with your own.

## 🤖 Models Evaluated

- YOLO11x
- YOLO26x
- RF-DETR

The models were compared based on:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Inference Performance

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- RF-DETR
- OpenCV
- NumPy
- WandB
- Albumentations

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/bee-monitoring-pipeline.git
cd bee-monitoring-pipeline
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Training

Example for YOLO

```bash
python train.py
```

Example for RF-DETR

```bash
python train_rfdetr.py
```

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix
- Precision-Recall Curve

---

## 📸 Example Results

(Add screenshots here)

| Ground Truth | Prediction |
|--------------|------------|
| Image | Image |

---

## 📁 Repository Structure

```
bee-monitoring-pipeline
│
├── configs
├── datasets
├── demo
├── docs
├── models
├── notebooks
├── results
├── scripts
├── src
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔬 Future Work

- Multi-object tracking
- Edge deployment (Jetson Orin Nano)
- Real-time inference
- Monitoring dashboard
- Pollinator behaviour analysis
- Species tracking across time

---

## 📚 References

- Bjerge et al., *Real-time Insect Tracking and Monitoring with Computer Vision and Deep Learning* (2022)
- Ratnayake et al., *Spatial Monitoring and Insect Behavioural Analysis Using Computer Vision for Precision Pollination* (2022)
- Chong et al., *A Computer Vision Dataset for Pollinator Detection under Real Field Conditions* (2025)

---

## 👨‍💻 Author

**Divyesh Kanani**

Master's Student – Mobile Robotics  
University of Bonn

GitHub: [link](https://github.com/divyesh-kanani)

LinkedIn: [link](www.linkedin.com/in/divyesh-kanani)

---

## 📄 License

This project is released under the MIT License.
