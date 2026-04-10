# SRS Parameters Dataset Directory

This directory contains extracted SRS parameters from experimental images, saved as `.npz` files.

## Contents

Each `.npz` file contains:
- `image_vec`: Filtered and normalized spectra from experimental images
- `noise_scale_vec`: Noise levels (std of silent region)
- `bg_scale_vec`: Background amplitudes
- `ratio_scale_vec`: Signal-to-noise ratios (SNR)
- `background`: Background spectrum
- `wavenumber_start`, `wavenumber_end`, `num_samples`: Metadata

## How to Generate

Run the main script:
```bash
python hsi_load_data.py
```

Edit the `SRS_CONFIGS` section in `hsi_load_data.py` to add your experimental directories.

## How to Load

### Option 1: Using HSI_Loader.load_srs_params() (Recommended)
```python
from hsi_load_data import HSI_Loader
import pandas as pd

background_df = pd.read_csv('water_HSI_76.csv')

# Load SRS parameters
loader, img_vec, noise_vec, bg_vec, ratio_vec = HSI_Loader.load_srs_params(
    'params_dataset/srs_params_61',
    background_df
)

# Use the parameters
print(f"Image spectra: {img_vec.shape}")
print(f"SNR range: [{ratio_vec.min():.2f}, {ratio_vec.max():.2f}]")
```

### Option 2: Direct numpy loading
```python
import numpy as np

# Load parameters directly
data = np.load('params_dataset/srs_params_61.npz')

noise_scale_vec = data['noise_scale_vec']
bg_scale_vec = data['bg_scale_vec']
ratio_scale_vec = data['ratio_scale_vec']
```

### Use with HSI_Labeled_Dataset
```python
from hsi_labeled_dataset import HSI_Labeled_Dataset

# The dataset class can load params directly
dataset = HSI_Labeled_Dataset.from_saved_params(
    params_file='params_dataset/srs_params_61.npz',
    mol_file='lipid_subtype_CH_61.csv',
    num_samples_per_class=2000
)
```
