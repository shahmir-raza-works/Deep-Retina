"""
Streamlit app — Diabetic Retinopathy Severity Grading
Loads the trained EfficientNetB0 classifier and predicts DR severity
(0-4) from an uploaded retina fundus image.

Run locally with:  streamlit run app.py
"""

import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# ---------------------------------------------------------------------------
# Config — must match the values used during training in the notebook
# ---------------------------------------------------------------------------
IMG_SIZE = 224  # matches the EfficientNetB0 classifier's input size
MODEL_PATH = "dr_model.keras"
LABEL_MAP_PATH = "label_map.json"

st.set_page_config(page_title="DR Severity Grading", page_icon="🩺", layout="centered")


# ---------------------------------------------------------------------------
# Preprocessing — identical to the notebook's ben_graham_preprocess pipeline
# ---------------------------------------------------------------------------
def crop_image_from_gray(img, tol=7):
    """Crop away the black border surrounding the retina."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    if mask.sum() == 0:  # image is (near) fully black — return as-is
        return img
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y0, y1 = ys[0], ys[-1]
    x0, x1 = xs[0], xs[-1]
    if x1 - x0 < 10 or y1 - y0 < 10:  # crop degenerate, bail out
        return img
    return img[y0:y1, x0:x1]


def ben_graham_preprocess(img, sigma_x=10):
    """Crop borders, resize, then apply local-contrast enhancement."""
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img


def preprocess_for_model(pil_image: Image.Image) -> np.ndarray:
    """PIL image (any mode/size) -> normalized (1, IMG_SIZE, IMG_SIZE, 3) float32 batch."""
    img_rgb = np.array(pil_image.convert("RGB"))
    processed = ben_graham_preprocess(img_rgb)
    processed = processed.astype(np.float32) / 255.0
    return np.expand_dims(processed, axis=0)


# ---------------------------------------------------------------------------
# Model + artifacts — cached so they only load once per session
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)

    with open(LABEL_MAP_PATH) as f:
        raw_label_map = json.load(f)
    label_map = {int(k): v for k, v in raw_label_map.items()}

    return model, label_map


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🩺 Diabetic Retinopathy Severity Grading")
st.write(
    "Upload a retinal fundus photograph and this model will estimate the diabetic "
    "retinopathy (DR) severity grade, trained on the APTOS 2019 Blindness Detection dataset."
)

st.warning(
    "⚠️ **This is an educational capstone project, not a medical device.** "
    "It has not been clinically validated and must not be used for real diagnosis or "
    "treatment decisions. Always consult an ophthalmologist for an actual diagnosis.",
    icon="⚠️",
)

uploaded_file = st.file_uploader(
    "Upload a fundus image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        model, label_map = load_artifacts()
        input_batch = preprocess_for_model(image)
        probabilities = model.predict(input_batch, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

    with col2:
        st.subheader("Prediction")
        st.metric(label="Severity Grade", value=f"{predicted_class} — {label_map[predicted_class]}")
        st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

    st.divider()
    st.subheader("Probability by Grade")
    st.bar_chart(
        {label_map[i]: float(probabilities[i]) for i in range(len(label_map))}
    )

    st.divider()
    st.subheader("Severity Scale Reference")
    st.table(
        {
            "Grade": list(label_map.keys()),
            "Meaning": list(label_map.values()),
        }
    )
else:
    st.info("Upload an image above to get a prediction.")

st.divider()
st.caption(
    "Model: EfficientNetB0 (transfer learning, 5-class classification) · "
    "Dataset: APTOS 2019 Blindness Detection · NAVTTC AI Capstone Project"
)
