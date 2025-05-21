# QSARPlasmaTissueCattle

# Development of Machine Learning and Chemical Language Model-Based QSAR Models for Predicting Drug Residue Depletion Half-Lives in Plasma and Tissues of Cattle Across Various Administration Routes

## 📌 Overview

This repository provides two distinct yet complementary pipelines for predicting the depletion half-lives (`LambdaZHl`) of veterinary drugs in cattle, encompassing both traditional machine learning (ML-QSAR) and advanced transformer-based approaches (ImprovedChemBERTa). Each section includes support for model interpretability, applicability domain analysis, and performance evaluation, facilitating robust and explainable predictions for food safety risk assessment.

---

## 🧪 Section I: Traditional ML-QSAR Pipeline

### 🔍 Description

This pipeline implements conventional QSAR modeling using curated molecular descriptors and regression algorithms. It supports descriptor calculation, feature selection, hyperparameter tuning, and SHAP-based interpretability.

### 📁 Repository Contents

| File Name                        | Description |
|----------------------------------|-------------|
| `ML_QSAR.py`                    | Full pipeline script including preprocessing, descriptor calculation, model training, evaluation, and interpretability. |
| `ML_QSAR.ipynb`                 | Jupyter notebook version for interactive exploration. |
| `Curated dataset.xlsx`         | Cleaned dataset with CAS numbers and LambdaZHl values. |
| `highest_lambdaZHl_with_CAS.xlsx` | Raw curated dataset with additional metadata. |
| `selected_features.xlsx`       | Selected features using F-statistics. |
| `best_model_params.xlsx`       | Optimized hyperparameters via Bayesian tuning. |
| `results_dict.xlsx`            | Model performance summary (R², RMSE). |

### 🔬 Key Features

- **Descriptors**: RDKit, MACCS, ECFP6, FCFP6
- **Models**: KNN, RF, SVR, DNN
- **Feature Selection**: `SelectKBest` (F-statistics)
- **Hyperparameter Optimization**: `BayesSearchCV`, `KerasTuner`
- **Interpretability**: SHAP values for top molecular features
- **Applicability Domain**: Williams plot (leverage vs. standardized residuals)

### ▶️ Usage

#### Setup Environment

```bash
pip install rdkit-pypi pubchempy scikit-learn scikit-optimize shap keras-tuner tensorflow
```

#### Run Pipeline

```bash
python ML_QSAR.py
```

Or explore via notebook:

```bash
ML_QSAR.ipynb
```

### 📊 Output

- CSVs of selected features and tuned parameters
- Keras model files for DNNs
- Visualizations: SHAP plots, correlation heatmaps, applicability domain

---

## 🤖 Section II: ImprovedChemBERTa Transformer QSAR Pipeline

### 🔍 Description

This section leverages a transformer-based architecture built on `ChemBERTa-77M-MTR`, augmented with route and tissue-specific metadata embeddings, stereochemistry tokens, and SMILES augmentation. It includes ensemble learning and DeepSHAP interpretation for robust predictions and insight.

### 📁 Repository Contents

| File Name                    | Description |
|------------------------------|-------------|
| `improvedchemberta.py`      | PyTorch-based ChemBERTa pipeline with training, evaluation, and SHAP analysis. |
| `ImprovedChemberta.ipynb`   | Notebook version for visualization and debugging. |
| `highest_lambdaZHl_with_CAS.xlsx` | Input dataset including SMILES, CAS, and metadata. |

### 🚀 Model Highlights

- **Backbone**: `DeepChem/ChemBERTa-77M-MTR`
- **Metadata Fusion**: Route and tissue embeddings
- **Stereo Tokens**: Integration of `@`, `/`, and `\` symbols
- **Training Strategy**:
  - Two-stage fine-tuning (frozen → full model)
  - Selective layer unfreezing
- **Augmentation**: Randomized SMILES strings
- **Optimization**: Optuna-based hyperparameter tuning
- **Ensemble Learning**: 5-fold independent model ensemble
- **Interpretability**: DeepSHAP token-level and embedding-level analysis
- **Applicability Domain**: Residual-leverage plots

### ▶️ Usage

#### Setup Environment

```bash
pip install shap optuna torch transformers rdkit scikit-learn matplotlib
```

#### Run Model

```bash
python improvedchemberta.py
```

Or interactively run:

```bash
ImprovedChemberta.ipynb
```

### 📊 Output

- Ensemble metrics: R², MAE, RMSE (test set)
- Token importance plots and SHAP bar charts
- Correlation, residual, and applicability domain diagnostics

### 🧬 Sample Output

```text
Ensemble Test R²: 0.72
Ensemble Test MAE: 0.49
Ensemble Test RMSE: 0.65
```

---

## 📚 Citation

If you use this repository, please cite:

> **Zhang Z, Tell LA, Lin Z.** (2025). *Development of Machine Learning and Chemical Language Model-Based QSAR Models for Predicting Drug Residue Depletion Half-Lives in Plasma and Tissues of Cattle Across Various Administration Routes* (under review)

---

## ✉️ Contact

For questions or collaborations, please contact:

- **Zhicheng Zhang** – [zhichengzhang@ufl.edu]  
- **Zhoumeng Lin** – [linzhoumeng@ufl.edu]
