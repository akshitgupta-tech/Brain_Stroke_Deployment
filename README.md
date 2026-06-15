# 🧠 Facial Paralysis Detection — Brain Stroke Classifier

A deep learning web app that detects **facial paralysis caused by stroke** from a face image, built with ResNet50 and deployed via Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.12.1-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red?logo=streamlit)
---

## 🖥️ Live Demo

🔗 [brainstrokedeployment-8xw9ujbq7ziz4qoekkapnn.streamlit.app](https://brainstrokedeployment-8xw9ujbq7ziz4qoekkapnn.streamlit.app)

---

# Facial Paralysis Detection via Cross-Modal Attention and Asymmetry-Aware Learning

> A hybrid MobileNetV3-Small and GCNN framework for automated facial paralysis detection — submitted as part of PCS 220 Multimedia Processing, TIET Patiala (2025).

---

## Overview

Facial paralysis is a critical early indicator of acute stroke. This project presents a hybrid deep learning framework that combines **MobileNetV3-Small** (for visual feature extraction) with a **Graph Convolutional Neural Network (GCNN)** (for geometric/structural analysis of facial landmarks), fused via a **cross-modal attention mechanism**.

The model simultaneously classifies facial paralysis (stroke vs. non-stroke) and estimates the degree of facial asymmetry through multi-task learning.

---

## Key Features

- **Dual-branch architecture** — MobileNetV3-Small handles appearance; GCNN handles landmark geometry
- **Cross-modal attention** — dynamically fuses visual and structural features
- **Asymmetry-aware loss** — gives higher weight to clinically significant asymmetric cases
- **Attention-based graph pooling** — focuses on the most diagnostically relevant facial nodes
- **5-fold cross-validation** — ensures robust, generalizable evaluation

---

## Architecture

```
Input Image
    │
    ├── Landmark Detection (dlib, 68 keypoints)
    │       └── Graph Construction G=(V,E)
    │               └── GCN Branch → Graph Feature Vector (256-d)
    │
    └── Image Tensor (224×224)
            └── MobileNetV3-Small + CBAM → Image Feature Vector (256-d)
                        │
                Cross-Modal Attention Layer
                        │
                Classification Head  +  Asymmetry Regression Head
                        │
                    Output (Stroke / No Stroke)
```

---

## Dataset

**FIASNAS** — 3,749 facial images covering both stroke and non-stroke cases, with variability in lighting, orientation, expression, and background.

| Split | Size |
|---|---|
| Train + Validation | 90% |
| Hold-out Test | 10% |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| CNN (baseline) | 0.6741 ± 0.0104 | 0.5568 ± 0.1544 | 0.6741 ± 0.0104 | 0.5537 ± 0.0290 |
| GCN (standalone) | 0.6691 ± 0.0246 | 0.4611 ± 0.2407 | 0.3196 ± 0.2722 | 0.3266 ± 0.2089 |
| MobileNetV3-Small | 0.9692 ± 0.0226 | 0.9206 ± 0.0526 | 0.9973 ± 0.0022 | 0.9566 ± 0.0298 |
| **Proposed (Hybrid)** | **0.9899 ± 0.0203** | **1.0000 ± 0.0000** | **0.9696 ± 0.0608** | **0.9835 ± 0.0329** |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| K-Folds | 5 |
| Dropout Rate | 0.3 |
| Hidden Dimension | 128 |
| Output Dimension | 256 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Epochs | 20 |

---

## Methodology

### Preprocessing
1. Face detection and background removal
2. Crop and resize to 224×224
3. CLAHE contrast enhancement
4. Eye-line alignment (corrects head tilt)
5. Pixel normalization

### MobileNetV3-Small Branch
- 13-layer backbone with Inverted Residual Blocks (IRB)
- CBAM (Convolutional Block Attention Module) applied after each block group
- Global Average Pooling → FC layers with Hardswish activation → 256-d embedding

### GCN Branch
- 68 facial landmarks extracted via **dlib**
- Graph G=(V,E): landmarks as nodes, anatomical connections as edges
- 2-layer GCN with BatchNorm, ReLU, Dropout (p=0.3)
- Attention-based graph pooling (2-layer MLP scoring per node) → 256-d graph embedding

### Fusion
- Cross-modal attention: image features as Query, graph features as Key & Value
- Attended features added back to image features (residual)
- Fused representation fed to classification + regression heads

### Training
- **Classification loss:** Binary Cross-Entropy (BCE)
- **Regression loss:** Mean Squared Error (MSE)
- **Total loss:** L_total = L_cls + λ·L_reg
- Optimizer: AdamW with early stopping; best model per fold saved

---

