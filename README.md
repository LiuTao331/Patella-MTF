# Patella-MTF

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for **"A multimodal Transformer fusion model for accurate sex estimation from patellar CT"**.

## Overview

This repository contains the complete code, pre‑trained models, and visualization tools to reproduce the experiments described in the paper. The project implements:

- **Data preprocessing** – DICOM anonymization, NIfTI conversion, isotropic resampling, bone‑window normalization.
- **Seven sex‑estimation models**:
  - Conventional morphometric logistic regression
  - Radiomics‑based logistic regression
  - 2D ResNet‑50 (MIP projections)
  - 3D ResNet‑50 (MedicalNet backbone)
  - Feature‑concatenation MLP
  - Weighted ensemble (probability averaging)
  - Transformer‑based multimodal fusion (the proposed method)
- **Extensive visualization** – cohort demographics, feature separation, model performance comparisons, and patellar geometry plots.

## Data Availability

- **Raw DICOM data** cannot be shared publicly due to patient privacy restrictions. Researchers may request access by contacting the corresponding author (liutao18328251625@163.com) and signing a data use agreement.
- **Pre‑trained model weights** are included in the `models/` directory for direct reproducibility of the paper's results.

## Repository Structure

```text
Patella-MTF/
├── models/                       # Pre‑trained model files
│   ├── conventional/
│   │   └── logistic_regression_pipeline.pkl
│   ├── radiomics/
│   │   ├── scaler_56feat.pkl
│   │   ├── lasso_selector_56feat.pkl
│   │   └── radiomics_logistic_model.pkl
│   ├── 2dcnn/
│   │   ├── model_state.pth
│   │   └── full_model.pth
│   ├── 3dcnn/
│   │   └── model_state.pth
│   ├── ensemble/
│   │   └── ensemble_predictor.pkl
│   ├── feature_concat/
│   │   └── feature_concat_mlp.pkl
│   └── transformer/
│       └── transformer_fusion.pkl
├── visualization/                # Figure‑generation scripts
│   ├── fig2_data_distribution.R
│   ├── fig3_patellar_morphology.R
│   ├── fig4_cnn_importance.R
│   ├── fig4_cross_modal_correlation.py
│   ├── fig4_lda_pca_visualization.py
│   ├── fig4_radiomics_features.R
│   ├── fig4i_fusion_gain_heatmap.R
│   ├── fig5_6_model_comparison.py
│   └── fig8_patella_geometry.R
├── anonymizer.py                 # DICOM de‑identification module
├── run_anonymization.py          # CLI for batch anonymization
├── preprocess.py                 # DICOM → NIfTI → isotropic resampling
├── train_Conventional.py
├── train_Radiomics.py
├── train_2DCNN.py
├── train_3DCNN.py
├── train_FeatureConcatMLP.py
├── train_Ensemble.py
├── train_TransformerFusion.py
├── requirements.txt
└── README.md
```

## Environment Setup

**Prerequisites**: Python ≥ 3.8, R ≥ 4.0, Git.

### Python environment

```bash
git clone https://github.com/LiuTao331/Patella-MTF.git
cd Patella-MTF

# Create and activate a conda environment (recommended)
conda create -n patella-mtf python=3.9
conda activate patella-mtf

# Install Python dependencies
pip install -r requirements.txt
```

**Important for 3D CNN**: Clone the MedicalNet repository and add it to your Python path. Verify the installation by importing the module:

```bash
git clone https://github.com/Tencent/MedicalNet.git

# Linux / Mac
export PYTHONPATH=$PWD/MedicalNet:$PYTHONPATH

# Windows
set PYTHONPATH=%cd%\MedicalNet;%PYTHONPATH%

# Check that the module can be imported
python -c "from models import resnet"
```

### R environment (for visualization)

```r
# Install all required R packages
install.packages(c("ggplot2", "dplyr", "readr", "tidyr", "patchwork", "optparse",
                   "ggpubr", "randomForest", "reshape2", "glmnet", "pheatmap",
                   "rgl", "Rvcg", "gridExtra", "geometry"))

# Rvcg might need to be installed from Bioconductor:
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("Rvcg")
```

> **Note**: `rgl` may require an X11 server on headless Linux systems (e.g., `Xvfb`). If you encounter display errors, consider running `Xvfb :99 &` before executing the script.

## Preprocessing Pipeline

The preprocessing scripts (`anonymizer.py`, `run_anonymization.py`, `preprocess.py`) assume that you have access to the original DICOM data. If you do not have the data, you may skip these steps and directly use the pre‑trained models for inference on your own preprocessed data.

### 1. DICOM Anonymization

```bash
python run_anonymization.py \
    /path/to/raw_dicom \
    /path/to/anonymized_dicom \
    --log-dir ./logs --verify-samples 5
```

### 2. DICOM to NIfTI Conversion & Preprocessing

```bash
# Convert DICOM series to NIfTI
python preprocess.py dicom2nifti \
    /path/to/anonymized_dicom \
    /path/to/nifti

# Resample to 0.5 mm isotropic and apply bone window
python preprocess.py preprocess \
    /path/to/nifti \
    /path/to/preprocessed \
    --spacing 0.5 0.5 0.5 --center 400 --width 1500
```

## Model Training & Inference

Each model training script accepts command‑line arguments. Use `python train_XXX.py --help` to see all options. Note that the training scripts expect the preprocessed data; if you are using your own dataset, adjust the paths accordingly. All scripts also accept `--help` for detailed argument descriptions.

**Example – Conventional model:**
```bash
python train_Conventional.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --train_measure data/train_measure.csv \
    --test_measure data/test_measure.csv \
    --trad_pipeline models/conventional/logistic_regression_pipeline.pkl \
    --output_dir models/conventional
```

**Example – Transformer Fusion:**
```bash
python train_TransformerFusion.py \
    --train_csv data/train.csv \
    --test_csv data/internal_test.csv \
    --external_csv data/external_test.csv \
    --train_measure data/train_measure.csv \
    ... \
    --output_dir models/transformer
```

*(Please refer to the help message of each script for all required arguments.)*

### Inference with pre‑trained models

To perform inference on new data, load the desired model and use the appropriate preprocessing pipeline. Below are minimal examples for three representative models.

**Conventional model:**
```python
import joblib
import pandas as pd

pipeline = joblib.load("models/conventional/logistic_regression_pipeline.pkl")
new_data = pd.read_csv("new_samples.csv")   # must contain the six morphometric columns
probabilities = pipeline.predict_proba(new_data)[:, 1]
predictions = probabilities >= 0.5
```

**Feature‑Concatenation MLP:**
```python
import joblib
artefact = joblib.load("models/feature_concat/feature_concat_mlp.pkl")
# artefact contains fold_states, scalers, PCA objects, etc.
# Refer to train_FeatureConcatMLP.py for the complete inference workflow.
```

**Transformer Fusion:**
```python
import joblib
artefact = joblib.load("models/transformer/transformer_fusion.pkl")
# artefact contains fold_states, preprocessing objects, and the decision threshold.
# Please see train_TransformerFusion.py for the feature extraction and prediction code.
```

For models requiring deep feature extraction (2D/3D CNN, Ensemble, Transformer), you will need to build the appropriate PyTorch dataset and feature extractor. The most reliable way is to adapt the inference section from the corresponding `train_*.py` script.

## Visualization

All figure‑generation scripts are located in the `visualization/` folder. Each script accepts `--help` to list available options.

```bash
# Python example (model comparison)
python visualization/fig5_6_model_comparison.py \
    --internal_csv data/internal_probs.csv \
    --external_csv data/external_probs.csv \
    --output_dir results/figures/fig5_6

# R example (data distribution)
Rscript visualization/fig2_data_distribution.R \
    --train_path data/train.csv \
    --test_path data/test.csv \
    --integrated_path data/integrated.xlsx \
    --external_path data/external.csv \
    --output_dir results/figures/fig2
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

The manuscript is currently under review. Once published, the citation details will be added here. In the meantime, if you use this code, please cite the corresponding paper or contact the authors. A BibTeX placeholder is provided below for your convenience.

```bibtex
@article{PatellaMTF,
  title   = {A multimodal Transformer fusion model for accurate sex estimation from patellar CT},
  author  = {Tao Liu and others},
  journal = {Under review},
  year    = {2025},
  note    = {Pre‑trained models and code available at https://github.com/LiuTao331/Patella-MTF}
}
```

## Contact

For questions or collaboration, please contact: **liutao18328251625@163.com**
