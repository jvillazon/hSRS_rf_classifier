# Molecule Dataset Directory

This directory contains processed and normalized molecular spectra saved as `.npz` files.

## Contents

Each `.npz` file contains:
- `normalized_molecules`: Normalized molecule spectra array (n_molecules+1, n_channels)
- `molecule_names`: Array of molecule names plus 'No Match'

## How to Generate

Run the main script:
```bash
python hsi_load_data.py
```

Edit the `MOLECULE_CONFIGS` section in `hsi_load_data.py` to add your datasets.

## How to Load

```python
from hsi_load_data import HSI_Loader
import pandas as pd

background_df = pd.read_csv('water_HSI_76.csv')
loader = HSI_Loader.load_from_file(
    'molecule_dataset/lipid_subtype_CH_61',
    background_df
)

# Access the data
molecules = loader.normalized_molecules
names = loader.molecule_names
```
