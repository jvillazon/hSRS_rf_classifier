"""
Train Random Forest (or SVM) classifiers for HSI lipid classification.

Usage (from workspace root):
    python -m training.sk_train
    python training/sk_train.py

Trained model is saved to models/ at the workspace root.
"""

import sys
import os

# Allow running as: python training/sk_train.py from the workspace root
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from helpers.hsi_labeled_dataset import HSI_Labeled_Dataset
from training.hsi_trainer import HSI_Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def main():

    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    _pkg_dir = _root  # workspace root

    # Dataset paths (bundled .npz files at workspace root)
    molecule_dataset_path = os.path.join(_pkg_dir, 'molecule_dataset', 'lipid_subtype_wn_61_test')
    srs_params_path       = os.path.join(_pkg_dir, 'params_dataset', 'srs_params_61')

    # Spectral parameters — must match the data and the target inference pipeline
    wn_1     = 2700
    wn_2     = 3100
    num_samp = 61

    # Model output
    model_name = 'rf_best_model' # Rename model name as necessary
    model_save_path = os.path.join(_pkg_dir, 'models', f'{model_name}.joblib')
    # ========================================================================

    # Build synthetic training dataset
    print("Building labeled training dataset...")
    dataset = HSI_Labeled_Dataset(
        molecule_dataset_path=molecule_dataset_path,
        srs_params_path=srs_params_path,
        num_samples_per_class=10000,
        normalize_per_molecule=False,
        compute_min_max=True,
        noise_multiplier=0.5,
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
        num_samples=num_samp,
    )

    # ── Random Forest (default) ────────────────────────────────────────────
    print("\nConfiguring Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=9,
        random_state=42,
        n_jobs=-1,
    )

    rf_trainer = HSI_Trainer(dataset, rf_model, model_type='sklearn')
    metrics = rf_trainer.train_sklearn_classifier(
        train_ratio=0.7,
        val_ratio=0.15,
        verbose=True,
        use_platt=True,
        sam_weighting=True,
        alpha=9,
    )

    print(f"\nTrain accuracy : {metrics['train_accuracy']:.4f}")
    print(f"Val   accuracy : {metrics['val_accuracy']:.4f}")

    eval_results = rf_trainer.evaluate()
    print(f"Test  accuracy : {eval_results['accuracy']:.4f}")
    print(eval_results['classification_report'])

    # Save model
    rf_trainer.save(model_save_path)

    # ── SVM baseline (optional — uncomment to run - NOT CURRENTLY SUPPORTED) ────────────────────────
    # svm_model = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    # svm_trainer = HSI_Trainer(dataset, svm_model, model_type='sklearn')
    # svm_trainer.train_sklearn_classifier(verbose=True)
    # svm_trainer.save(os.path.join(_pkg_dir, 'models', 'svm_best_model.joblib'))


if __name__ == '__main__':
    main()
