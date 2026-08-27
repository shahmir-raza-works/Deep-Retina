"""
Streamlit app — Diabetic Retinopathy Severity Grading
Loads the trained EfficientNetB0 classifier and predicts DR severity
(0-4) from an uploaded retina fundus image.

Run locally with:  streamlit run app.py
"""

import json

import cv2
import matplotlib.pyplot as plt
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

ACCENT = "#22e08a"
ACCENT_DIM = "#0f7a4d"
BG = "#05080a"
CARD_BG = "#0e1512"
BORDER = "rgba(255,255,255,0.07)"
BORDER_MPL = "#26302b"  # matplotlib doesn't accept CSS rgba() strings
TEXT_MAIN = "#eef4f1"
TEXT_DIM = "#8fa39a"

st.set_page_config(page_title="Deep Retina", page_icon="👁️", layout="wide")

# ---------------------------------------------------------------------------
# Global dark theme — cards, pills, accent colors
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 15% 10%, rgba(34,224,138,0.08), transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(34,224,138,0.05), transparent 35%),
            {BG};
        color: {TEXT_MAIN};
    }}

    section.main > div.block-container {{
        padding-top: 2rem;
        max-width: 1200px;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT_MAIN} !important;
        font-weight: 800 !important;
    }}

    p, span, label, .stMarkdown {{
        color: {TEXT_DIM};
    }}

    /* ---- Card containers (st.container(border=True)) ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, {CARD_BG}, #0a100d) !important;
        border: 1px solid {BORDER} !important;
        border-radius: 18px !important;
        padding: 6px 4px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}

    /* ---- File uploader ---- */
    [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.02) !important;
        border: 1.5px dashed {ACCENT_DIM} !important;
        border-radius: 14px !important;
    }}

    /* ---- Metric ---- */
    [data-testid="stMetricValue"] {{
        color: {ACCENT} !important;
        font-weight: 800 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_DIM} !important;
    }}

    /* ---- Progress bar ---- */
    div[data-testid="stProgress"] div[role="progressbar"] > div {{
        background-image: linear-gradient(90deg, {ACCENT_DIM}, {ACCENT}) !important;
    }}

    /* ---- Badge ---- */
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(34,224,138,0.12);
        color: {ACCENT};
        border: 1px solid rgba(34,224,138,0.3);
    }}

    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def preprocess_for_model(pil_image: Image.Image):
    """PIL image (any mode/size) -> (normalized (1, IMG_SIZE, IMG_SIZE, 3) float32 batch,
    uint8 preprocessed image for display)."""
    img_rgb = np.array(pil_image.convert("RGB"))
    processed = ben_graham_preprocess(img_rgb)  # uint8
    normalized = processed.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0), processed


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


def plot_probabilities(label_map: dict, probabilities: np.ndarray):
    """Dark-themed horizontal bar chart of per-class probability."""
    labels = [label_map[i] for i in range(len(label_map))]
    values = [float(p) for p in probabilities]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    bars = ax.barh(labels, values, color=ACCENT_DIM, height=0.55, zorder=3)
    top_idx = int(np.argmax(values))
    bars[top_idx].set_color(ACCENT)

    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Probability", color=TEXT_DIM, fontsize=9)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=BORDER_MPL, linewidth=0.6, zorder=0)

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.9), bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%", va="center", color=TEXT_MAIN, fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_confidence_donut(confidence: float, predicted_label: str):
    """Dark-themed donut chart highlighting model confidence for the top prediction."""
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    fig.patch.set_facecolor(CARD_BG)

    sizes = [confidence, 1 - confidence]
    colors = [ACCENT, "#1b2420"]
    ax.pie(
        sizes, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor=CARD_BG),
    )
    ax.text(0, 0.08, f"{confidence * 100:.0f}%", ha="center", va="center",
             color=TEXT_MAIN, fontsize=22, fontweight="bold")
    ax.text(0, -0.22, predicted_label, ha="center", va="center", color=TEXT_DIM, fontsize=9)
    ax.set(aspect="equal")
    fig.tight_layout()
    return fig


def plot_probability_radar(label_map: dict, probabilities: np.ndarray):
    """Radar/polar chart showing probability spread across all 5 severity grades."""
    labels = [label_map[i] for i in range(len(label_map))]
    values = [float(p) for p in probabilities]
    values += values[:1]  # close the loop

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(4.2, 4.2))
    fig.patch.set_facecolor(CARD_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(CARD_BG)

    ax.plot(angles, values, color=ACCENT, linewidth=2)
    ax.fill(angles, values, color=ACCENT, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_DIM, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color(BORDER_MPL)
    ax.grid(color=BORDER_MPL, linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_rgb_histogram(img_uint8: np.ndarray):
    """Pixel-intensity histogram (RGB channels) of the preprocessed image fed to the model."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    channel_colors = {"Red": "#ff6b6b", "Green": "#51cf66", "Blue": "#4dabf7"}
    for i, (name, color) in enumerate(channel_colors.items()):
        ax.hist(img_uint8[..., i].ravel(), bins=40, color=color, alpha=0.55, label=name)

    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel value", color=TEXT_DIM, fontsize=9)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=BORDER_MPL, linewidth=0.5)
    legend = ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_MPL)
    for text in legend.get_texts():
        text.set_color(TEXT_MAIN)
    fig.tight_layout()
    return fig
# ---------------------------------------------------------------------------
st.markdown("## Deep <👁️> Retina")
st.markdown(
    f'<span class="badge">EfficientNetB0 · APTOS 2019</span>&nbsp;&nbsp;'
    f'<span style="color:{TEXT_DIM}; font-size:14px;">Diabetic retinopathy severity grading from a fundus photo</span>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Tabs — Scan (upload + result) / Prediction Insights (all plots for this prediction)
# ---------------------------------------------------------------------------
scan_tab, insights_tab = st.tabs(["🔍 Scan", "📊 Prediction Insights"])

with scan_tab:
    left, right = st.columns([2, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("#### Upload a Fundus Image")
            uploaded_file = st.file_uploader(
                "Upload a fundus image (JPG or PNG)", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_col, result_col = st.columns([1, 1], gap="medium")

                with img_col:
                    st.image(image, caption="Uploaded image", use_container_width=True)

                with st.spinner("Analyzing image..."):
                    model, label_map = load_artifacts()
                    input_batch, processed_img = preprocess_for_model(image)
                    probabilities = model.predict(input_batch, verbose=0)[0]
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_class])

                with result_col:
                    st.metric(label="Severity Grade", value=f"{predicted_class} — {label_map[predicted_class]}")
                    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

                st.markdown("###### Probability by Grade")
                st.pyplot(plot_probabilities(label_map, probabilities), use_container_width=True)
            else:
                st.info("Upload an image above to get a prediction.")

    with right:
        with st.container(border=True):
            st.markdown("#### 🧠 Model Details")
            st.markdown(
                f"""
                <div style="line-height: 2;">
                <span class="badge">Architecture</span> EfficientNetB0 (transfer learning)<br>
                <span class="badge">Dataset</span> APTOS 2019 Blindness Detection<br>
                <span class="badge">Classes</span> 5 severity grades (0–4)
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### Severity Scale")
            scale_html = ""
            for grade, label in {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}.items():
                scale_html += (
                    f'<div style="display:flex; justify-content:space-between; padding:6px 0; '
                    f'border-bottom:1px solid {BORDER};">'
                    f'<span style="color:{TEXT_MAIN}; font-weight:600;">{grade}</span>'
                    f'<span style="color:{TEXT_DIM};">{label}</span></div>'
                )
            st.markdown(scale_html, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### ⚠️ Disclaimer")
            st.markdown(
                f'<span style="color:{TEXT_DIM}; font-size:13px;">'
                "This is an educational capstone project, not a medical device. It has not been "
                "clinically validated and must not be used for real diagnosis or treatment decisions. "
                "Always consult an ophthalmologist for an actual diagnosis."
                "</span>",
                unsafe_allow_html=True,
            )

with insights_tab:
    if uploaded_file is None:
        st.info("Upload an image in the **Scan** tab first to see prediction plots here.")
    else:
        with st.container(border=True):
            st.markdown("#### Confidence & Probability Breakdown")
            c1, c2 = st.columns([1, 1.6], gap="medium")
            with c1:
                st.pyplot(plot_confidence_donut(confidence, label_map[predicted_class]), use_container_width=True)
            with c2:
                st.pyplot(plot_probabilities(label_map, probabilities), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Severity Probability Radar")
            st.pyplot(plot_probability_radar(label_map, probabilities), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Preprocessing: Original vs. Model Input")
            colA, colB = st.columns(2, gap="medium")
            with colA:
                st.image(image, caption="Original upload", use_container_width=True)
            with colB:
                st.image(processed_img, caption="After crop + Ben Graham enhancement", use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Pixel Intensity Distribution (model input)")
            st.pyplot(plot_rgb_histogram(processed_img), use_container_width=True)

st.write("")
st.markdown(
    f'<div style="text-align:center; color:{TEXT_DIM}; font-size:12px; padding-top: 12px;">'
    "NAVTTC AI (ML/DL) Capstone Project · Made by Muhammad Shahmir Raza"
    "</div>",
    unsafe_allow_html=True,
)
