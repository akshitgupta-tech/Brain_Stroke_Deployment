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

## 🔍 What It Does

- Upload a **face photograph**
- The app runs it through a fine-tuned **ResNet50** model
- Outputs: **Stroke ⚠️** or **Non-Stroke ✅** with confidence score and raw probability
- Supports **batch prediction** — upload many images at once and export results as CSV

---

## 🧠 Model

| Detail | Value |
|---|---|
| Architecture | ResNet50 (ImageNet pretrained) |
| Training strategy | 2-phase: Transfer Learning → Fine-tuning |
| Validation | 5-Fold Stratified Cross-Validation |
| Input size | 224 × 224 RGB |
| Preprocessing | `keras.applications.resnet50.preprocess_input` |
| Classes | Non-Stroke (0) · Stroke (1) |
| Output | Sigmoid probability |
| Model format | `.keras` (Keras 3 native format) |

The model was trained with class balancing (oversampling), real-time augmentation (flips, rotations, zoom), and class-weighted loss to handle dataset imbalance.

---

## 📁 Project Structure

```
Brain_Stroke_Deployment/
├── app.py               # Streamlit application
├── model.keras          # Trained ResNet50 model (Keras 3 format)
├── requirements.txt     # Runtime dependencies
└── README.md            # This file
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/akshitgupta-tech/Brain_Stroke_Deployment.git
cd Brain_Stroke_Deployment
```

**2. Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | >=1.40.0 | Web interface |
| `tensorflow` | 2.21.0 | Model backend |
| `keras` | 3.12.1 | ResNet50 & inference |
| `opencv-python-headless` | latest | Image preprocessing |
| `Pillow` | latest | Image handling |
| `numpy` | latest | Array operations |
| `pandas` | latest | Batch results & CSV export |

---

## 🗂️ Training

The model was trained separately using `resnet50_train.py`. Training details:

- **Dataset:** Stroke / Non-Stroke facial images
- **Augmentation:** Horizontal flip, rotation (±10°), zoom (0.9–1.1×)
- **Phase 1:** Frozen ResNet50 backbone, 15 epochs, LR = 1e-4
- **Phase 2:** Top 30 layers unfrozen, 20 epochs, LR = 1e-5
- **Early stopping** with best-weight restoration
- **Best fold model** saved and converted to `.keras` format for deployment

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is **not a medical device** and should not be used for clinical diagnosis. Always consult a qualified medical professional for any health concerns.

