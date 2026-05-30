# Facial Expression Recognition (FER) Using Deep Convolutional Neural Networks

An academic-grade, modular computer vision pipeline implemented in TensorFlow/Keras for static Facial Expression Recognition (FER). This repository transitions a verified Master’s thesis project from interactive Jupyter notebooks into high-performance, deployable Python modules optimized for evaluation on unconstrained environments.

## 📊 Empirical Performance Milestones
* **CK+ Dataset Variant:** Achieved **100.00%** Test Classification Accuracy using an optimized deep sequential architecture.
* **FER2013 Dataset Variant:** Achieved **73.61%** Test Validation Accuracy on unconstrained "in-the-wild" images by combining dual-convolutional blocks, heavy dropout regularization, and targeted data balancing techniques.

---

## 🛠️ Methodological & Architectural Design

### 1. Advanced Data Processing & Resampling Pipeline
Real-world expressional datasets suffer heavily from highly skewed class distributions (e.g., severe under-representation of the *Disgust* category in FER2013). To prevent optimization bias, this pipeline applies a strategic two-stage resampling routine:
* **Synthetic Neighborhood Balancing:** Leverages class oversampling and undersampling boundaries via `imbalanced-learn` to normalize baseline class assignments and maximize backpropagation efficiency.
* **Robust Spatial Augmentation:** Employs an on-the-fly random image transformation engine to introduce spatial variations, protecting dense classification paths from co-adaptation and overfitting traps.

### 2. Network Layout Configurations
* **`CKPlus_Sequential_DCNN`:** Built using a progressively deepening micro-kernel structure (filters scaling from 32 up to 512) interleaved with Batch Normalization layers and strict dropout margins ($0.25$).
* **`FER2013_DCNN` (Functional API):** Features stacked dual-convolutional sequences equipped with large $5\times5$ receptive fields in initial blocks, followed by deep $3\times3$ spatial abstraction layers and intensive dropout regularizations ($0.4$ to $0.5$).

---

## 📂 Repository Root & Structure

```text
Facial-Expression-Recognition-using-DCNN/
│
├── models/
│   ├── __init__.py
│   ├── ck_model.py          # Sequential architecture definition for CK+
│   └── fer2013_model.py     # Functional architecture definition for FER2013
│
├── train_ckplus.py          # Executable training pipeline for CK+
├── train_fer2013.py         # Executable training pipeline for FER2013
├── requirements.txt         # Production library dependencies
├── .gitignore               # Safe exclusion rules for local dataset noise
└── README.md                # Project landing documentation


## 📜 Academic Reference & Citation

If you use this code or find the research parameters helpful in your work, please reference the original Master's thesis:

```text
Title:        Facial Expressions Recognition using Deep Convolutional Neural Networks
Degree:       Master of Sciences in Electrical Engineering (Specialization: Control)
Institution:  University of Tabriz, Faculty of Electrical and Computer Engineering
Author:       Paniz Hosseinpour Ardeshiri
Date:         September 2022
Metrics:      73.61% Accuracy (FER2013) | 100.00% Accuracy (CK+)