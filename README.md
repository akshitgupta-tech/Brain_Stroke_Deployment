# 🧠 Facial Paralysis Detection — Brain Stroke Classifier

A deep learning web app that detects **facial paralysis caused by stroke** from a face image, built with ResNet50 and deployed via Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖥️ Live Demo

> 🔗 [Click here to open the app](https://YOUR_USERNAME-brain-stroke-deployment-app-xxxx.streamlit.app)  
> *(Replace with your actual Streamlit URL after deployment)*

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

The model was trained with class balancing (oversampling), real-time augmentation (flips, rotations, zoom), and class-weighted loss to handle dataset imbalance.

---

## 📁 Project Structure

```
Brain_Stroke_Deployment/
├── app.py               # Streamlit application
├── model.h5             # Trained ResNet50 weights (Git LFS)
├── requirements.txt     # Runtime dependencies
└── README.md            # This file
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/Brain_Stroke_Deployment.git
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

| Package | Purpose |
|---|---|
| `streamlit` | Web interface |
| `tensorflow` | ResNet50 model & inference |
| `opencv-python-headless` | Image loading & preprocessing |
| `Pillow` | Image handling |
| `numpy` | Array operations |
| `pandas` | Batch results & CSV export |

---

## 🗂️ Training

The model was trained separately using `train.py`. Training details:

- **Dataset:** Stroke / Non-Stroke facial images
- **Augmentation:** Horizontal flip, rotation (±10°), zoom (0.9–1.1×)
- **Phase 1:** Frozen ResNet50 backbone, 15 epochs, LR = 1e-4
- **Phase 2:** Top 30 layers unfrozen, 20 epochs, LR = 1e-5
- **Early stopping** with best-weight restoration

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is **not a medical device** and should not be used for clinical diagnosis. Always consult a qualified medical professional for any health concerns.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
