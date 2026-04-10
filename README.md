# HSI Random Forest Lipid Classifier

Classifies pixels in stimulated Raman scattering (SRS) hyperspectral images into lipid species categories using a pre-trained Random Forest model.

Developed in the Laboratory of Optical Bioimaging and Spectroscopy under Dr. Lingyan Shi @ UCSD.

---

## Repository Structure

```
packaged_rf_classifier/
│
│  ── INFERENCE ──────────────────────────────────────────────
├── classify.py               ★  Run this to classify your images
│
├── models/                   Pre-trained classifier
│   └── rf_n150_md40_mss2_msl9_a9_best_model_platt.joblib
│
├── molecule_dataset/         Bundled reference lipid spectra (.npz)
├── params_dataset/           Bundled SRS background parameters (.npz)
│
│  ── INFERENCE LIBRARY (internal — no need to edit) ─────────
├── helpers/
│   ├── hsi_sk_classifier.py      Classifier: inference, SHAP, SAM weighting
│   ├── hsi_unlabeled_dataset.py  Loads .tif image stacks
│   ├── hsi_labeled_dataset.py    Reference spectra for SAM weighting
│   ├── hsi_normalization.py      Spectral background subtraction
│   └── hsi_visualizer.py         Writes prediction CSVs and probability TIFFs
│
│  ── TRAINING (optional) ─────────────────────────────────────
├── training/
│   ├── sk_train.py           ★  Run this to retrain the model
│   ├── hsi_trainer.py            Training loop (sklearn + PyTorch)
│   ├── hsi_load_data.py          Prepares molecule datasets and SRS params
│   └── training_data/            Raw spectra for building training datasets
│       ├── averaged_spectra.csv
│       └── water_HSI_76.csv
│
│  ── ENVIRONMENT ─────────────────────────────────────────────
├── environment.yml           Conda environment (recommended)
└── requirements.txt          pip requirements
```

---

## Environment Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate hsi_rf_classifier
```

### pip

```bash
pip install -r requirements.txt

# Then install PyTorch for your platform:
pip install torch --index-url https://download.pytorch.org/whl/cpu        # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121      # CUDA 12.1
# See https://pytorch.org/get-started/locally/ for all options
```

---

## Running Inference

### Step 1 — Set your data path

Open `classify.py` and find the **USER CONFIG** block:

```python
# ========================================================================
# USER CONFIG — edit these values for your environment
# ========================================================================
base_directory = None     # ← set to your .tif image folder, e.g.:
                          #   r"D:\my_project\registered_data\data"
```

### Step 2 — Run

```bash
conda activate hsi_rf_classifier
python classify.py
```

Outputs are saved alongside your images:

```
<parent of base_directory>/
└── rf_n150_..._outputs/
    └── <image_name>/
        ├── <image_name>_predictions.csv
        └── <image_name>_probabilities.tif
```

---

## (Optional) Retraining

```bash
python training/sk_train.py
# → Saves model to: models/*.joblib
```

To regenerate training datasets from your own molecule spectra:

```bash
# Edit training/hsi_load_data.py — the CONFIGS block near the bottom
python training/hsi_load_data.py
# → molecule_dataset/   ← updated
# → params_dataset/     ← updated
```

---

## Bundled Model

| Parameter | Value |
|---|---|
| Type | Random Forest + Platt calibration |
| Estimators | 150 |
| Max depth | 40 |
| Min samples leaf | 9 |
| Input | 61 channels · 2700–3100 cm⁻¹ |
| Output | ~30 lipid classes + background |

---

## Citation

If you use this pipeline in published work, please cite the Laboratory of Optical Bioimaging and Spectroscopy, UCSD (Dr. Lingyan Shi).
