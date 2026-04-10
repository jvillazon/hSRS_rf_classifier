"""
HSI Lipid Classifier — inference entry point.

Usage:
    python classify.py

Outputs per-image prediction CSVs and probability TIFFs to <output_base>/.
Edit the USER CONFIG block below to point at your data and model.
"""

import os

from helpers.hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from helpers.hsi_visualizer import HSI_Visualizer
from helpers.hsi_sk_classifier import HSI_Classifier
from helpers.hsi_labeled_dataset import HSI_Labeled_Dataset


def main():
    # ========================================================================
    # USER CONFIG — edit these values for your environment
    # ========================================================================

    # Path to the folder containing your HSI .tif image stacks
    base_directory = r"D:\integrated_pipeline\HSI_data\data"     # e.g. r"D:\my_project\registered_data\data"

    # Spectral parameters — must match the bundled model and datasets
    wn_1     = 2700   # Starting wavenumber (cm⁻¹)
    wn_2     = 3100   # Ending wavenumber (cm⁻¹)
    num_samp = 61     # Number of spectral channels
    ch_start = int(((wn_2 - 2800) / (wn_2 - wn_1)) * num_samp)

    # Pre-trained model bundled in models/
    chosen_model_name = 'rf_n150_md40_mss2_msl9_a9_best_model_platt'
    model_path = os.path.join(os.path.dirname(__file__), 'models',
                              f'{chosen_model_name}.joblib')

    # ========================================================================

    if base_directory is None:
        raise ValueError(
            "base_directory is not set.\n"
            "Open classify.py and set base_directory to your .tif image folder "
            "in the USER CONFIG block."
        )

    _pkg = os.path.dirname(__file__)

    # Labeled reference dataset — used for SAM spectral weighting
    labeled_dataset = HSI_Labeled_Dataset(
        molecule_dataset_path=os.path.join(_pkg, 'molecule_dataset', 'lipid_subtype_wn_61_test'),
        srs_params_path=os.path.join(_pkg, 'params_dataset', 'srs_params_61'),
        num_samples_per_class=10000,
        normalize_per_molecule=False,
        compute_min_max=True,
        noise_multiplier=0.5,
    )

    # Visualizer — writes prediction CSVs and probability TIFFs
    visualizer = HSI_Visualizer(
        mol_path=os.path.join(_pkg, 'molecule_dataset', 'lipid_subtype_wn_61_test.npz'),
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
        num_samples=num_samp,
    )

    # Unlabeled dataset — loads and normalizes .tif image stacks
    dataset = HSI_Unlabeled_Dataset(
        base_directory,
        ch_start,
        transform=None,
        image_normalization=True,
        min_max_normalization=False,
        num_samples=num_samp,
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
        compute_stats=True,
    )

    output_base = os.path.join(
        os.path.dirname(base_directory), f"{chosen_model_name}_outputs" # Output directory for subfolders
    )
    classifier = HSI_Classifier(
        dataset,
        model_path=model_path,
        output_base=output_base,
        visualizer=visualizer,
        labeled_dataset=labeled_dataset,
    )

    print("\nRunning Random Forest inference...")
    classifier.predict(alpha=10, generate_shap=True)


if __name__ == '__main__':
    main()
