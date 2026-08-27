# APTOS 2019 Blindness Detection — Diabetic Retinopathy Severity Classification

Capstone project for NAVTTC's Prime Minister's Hunarmand Pakistan Program — Artificial Intelligence
(Machine Learning & Deep Learning) course, Assignment 7.

A deep learning model that grades diabetic retinopathy (DR) severity from a retinal fundus photograph,
wrapped in a Streamlit web app so it can be tried out live.

## Problem Statement

Diabetic Retinopathy is a leading cause of preventable blindness. Grading its severity from a fundus
photograph currently requires a trained ophthalmologist, which limits how widely screening can scale,
especially in low-resource settings. This project trains a CNN to automatically predict DR severity from
an image, as a fast, low-cost screening/triage aid.

**Severity scale (ordinal, 0–4):**

| Grade | Meaning |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
(Kaggle / Asia Pacific Tele-Ophthalmology Society) — ~3,662 labeled retina images captured under varied
real-world clinical conditions (different cameras, lighting, image quality).

## Approach

1. **Preprocessing** — crop black borders (`crop_image_from_gray`), then apply Ben Graham-style local
   contrast enhancement to make small lesions (microaneurysms, hemorrhages) more visible.
2. **Modeling** — EfficientNetB3 (ImageNet-pretrained) as a feature extractor, with a single-unit
   **regression head** rather than 5-way softmax classification. DR severity is ordinal, and the official
   competition metric (Quadratic Weighted Kappa) rewards predictions that are numerically close to the
   true grade — regression captures that ordering; classification throws it away.
3. **Class imbalance** — the dataset skews heavily toward "No DR". Handled with sqrt-dampened per-sample
   loss weighting (softer than full inverse-frequency balancing, to avoid destabilizing the boundary
   between adjacent severity grades).
4. **Rounding** — an `OptimizedRounder` searches for the four cut-points that convert the model's
   continuous output into discrete class labels 0–4, maximizing validation QWK (instead of naively
   rounding at 0.5/1.5/2.5/3.5).
5. **Evaluation** — Quadratic Weighted Kappa (primary metric), confusion matrix, per-class
   precision/recall/F1.

## Repository Structure

```
.
├── Assignment7_YourName.ipynb   # Full pipeline: EDA, preprocessing, training, evaluation
├── app.py                       # Streamlit inference app
├── dr_model.keras               # Trained model weights (see note below)
├── label_map.json               # Class index -> label name
├── rounding_thresholds.json     # Learned OptimizedRounder cut-points
├── requirements.txt
└── README.md
```

> **Note on model weights:** `dr_model.keras` can be large. If it exceeds GitHub's file-size limits, host
> it externally (e.g. Hugging Face Hub, Google Drive) and update the loading path in `app.py`, or use
> [Git LFS](https://git-lfs.com/).

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

## Running the Notebook

The notebook expects the APTOS dataset at the Kaggle competition input paths:

```
/kaggle/input/competitions/aptos2019-blindness-detection/train_images
/kaggle/input/competitions/aptos2019-blindness-detection/test_images
/kaggle/input/competitions/aptos2019-blindness-detection/train.csv
/kaggle/input/competitions/aptos2019-blindness-detection/test.csv
/kaggle/input/competitions/aptos2019-blindness-detection/sample_submission.csv
```

Open and run `Assignment7_YourName.ipynb` on [Kaggle](https://www.kaggle.com/) (with GPU accelerator
enabled) or adapt the paths for a local copy of the dataset. Running it end to end produces
`dr_model.keras`, `label_map.json`, and `rounding_thresholds.json`.

## Running the Streamlit App Locally

```bash
streamlit run app.py
```

Upload a fundus image and the app returns the predicted DR severity grade with the model's confidence.

## Live Demo

- **Streamlit Cloud app:** _add your deployed link here_
- **LinkedIn post:** _add your post link here_

## Results

| Metric | Value |
|---|---|
| Quadratic Weighted Kappa (validation) | _fill in after training_ |
| Accuracy | _fill in after training (secondary metric — QWK is primary)_ |

See the notebook's Evaluation section for the full confusion matrix and per-class report.

## Tech Stack

- TensorFlow / Keras (EfficientNetB3 transfer learning)
- OpenCV (image preprocessing)
- scikit-learn / SciPy (metrics, threshold optimization)
- Streamlit (deployment)

## Author

_Your Name_ — NAVTTC Hunarmand Pakistan Program, AI (ML/DL) Course

## License

This project is released under the MIT License. The APTOS 2019 dataset is subject to its own
[competition rules and license](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules).
# Deep-Retina
