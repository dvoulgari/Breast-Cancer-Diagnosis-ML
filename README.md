# Breast Cancer Classification – MLCB Assignment 2

This repository contains my solution for Assignment #2 of the course **Machine Learning in Computational Biology**. The task involves building a robust ML pipeline to classify breast cancer tumors as benign or malignant using repeated nested cross-validation (rnCV).

---

## 📂 Structure

- `notebooks/` – EDA and evaluation workflows
- `src/` – Custom classes and functions (e.g., `ClassifierCV`, `ClassSpecificImputer`)
- `models/` – Saved final model (`.pkl`)
- `figures/` – Generated plots
- `*.csv/png` – Evaluation results and visualizations

---

## ⚙️ Highlights

- Preprocessing with class-specific imputation
- Evaluation via 10×5×3 rnCV
- Compared 6 classifiers including SVM, RF, LightGBM
- Bonus: Feature selection + class balancing (SMOTE, NearMiss, etc.)
