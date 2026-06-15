# -*- coding: utf-8 -*-
"""
Streamlit App — Facial Paralysis Detection (ResNet50)
Loads model.keras directly from the repo. No upload needed.
"""

import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Paralysis Detector",
    page_icon="🧠",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
LABELS   = {0: "Non-Stroke  ✅", 1: "Stroke  ⚠️"}
COLORS   = {0: "#2ecc71",        1: "#e74c3c"}

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)   # .keras has zero compat issues


def preprocess(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)


def predict(model, image: Image.Image):
    x    = preprocess(image)
    prob = float(model.predict(x, verbose=0)[0][0])
    idx  = int(prob >= 0.5)
    conf = prob if idx == 1 else 1.0 - prob
    return idx, conf, prob


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Facial Paralysis Detection")
st.markdown("Upload a face image to classify it as **Stroke** or **Non-Stroke**.")
st.divider()

# ── Auto-load model from repo ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model:\n\n{e}")
else:
    st.error(
        "`model.keras` not found in repo. "
        "Run `convert_model.py` locally and push `model.keras` to GitHub."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Status")
    if model is not None:
        st.success("Model loaded ✔")
        st.caption(f"Input shape: {model.input_shape}")
    else:
        st.error("No model loaded")

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        """
        **Architecture:** ResNet50 (ImageNet pretrained)  
        **Classes:** Non-Stroke · Stroke  
        **Input:** 224 × 224 RGB  
        **Preprocessing:** ResNet50 ImageNet normalisation
        """
    )

# ── Main panel ────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)

with col_result:
    st.subheader("🔍 Prediction")
    if not uploaded:
        st.info("Upload an image to get started.")
    elif model is None:
        st.warning("Model not loaded.")
    else:
        with st.spinner("Running inference …"):
            label_idx, confidence, raw_prob = predict(model, img)

        label = LABELS[label_idx]
        color = COLORS[label_idx]

        st.markdown(
            f"""
            <div style="
                background:{color}22;
                border:2px solid {color};
                border-radius:10px;
                padding:20px;
                text-align:center;
            ">
                <h2 style="color:{color}; margin:0">{label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.divider()
        st.markdown("**Raw probabilities**")
        st.progress(raw_prob, text=f"Stroke probability: {raw_prob:.3f}")

        with st.expander("Details"):
            st.write({
                "stroke_probability":     round(raw_prob, 4),
                "non_stroke_probability": round(1 - raw_prob, 4),
                "predicted_class":        "Stroke" if label_idx == 1 else "Non-Stroke",
                "threshold":              0.5,
            })

# ── Batch mode ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📂 Batch Prediction")

batch_files = st.file_uploader(
    "Upload multiple images", type=["jpg", "jpeg", "png"],
    accept_multiple_files=True, key="batch"
)

if batch_files:
    if model is None:
        st.warning("Model not loaded.")
    else:
        import pandas as pd
        results = []
        prog = st.progress(0, text="Processing …")

        for i, f in enumerate(batch_files):
            pil = Image.open(f)
            idx, conf, prob = predict(model, pil)
            results.append({
                "Filename":    f.name,
                "Prediction":  "Stroke" if idx == 1 else "Non-Stroke",
                "Confidence":  f"{conf * 100:.1f}%",
                "Stroke Prob": round(prob, 4),
            })
            prog.progress((i + 1) / len(batch_files), text=f"Processing {f.name} …")

        prog.empty()
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download results CSV", csv, "predictions.csv", "text/csv")