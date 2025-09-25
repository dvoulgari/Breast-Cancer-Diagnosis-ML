# Breast Cancer Diagnosis: A Comparative Analysis
This repository contains the solution for Assignment #2 of the Machine Learning in Computational Biology course. The project focuses on building and comparing predictive models for breast cancer diagnosis using a dataset of 512 tumor samples.

## Project Objective

The main goal of this assignment is to develop robust machine learning models to classify tumors as either benign or malignant based on a set of thirty features.

## Methodology

The notebook outlines a comprehensive and rigorous machine learning workflow:

1) Data Preprocessing: The project addresses data challenges by using a custom class-specific imputation method to handle missing values, ensuring data integrity before model training.

2) Model Comparison: Six different classifiers are trained and evaluated, including Support Vector Machine (SVM), Random Forest, and LightGBM, to identify the most effective model for this classification task.

3) Evaluation: A key highlight of this project is the use of a 10x5x3 repeated nested cross-validation (rnCV) approach. This method provides a highly reliable assessment of model performance and helps prevent overfitting by robustly estimating the model's generalization error.

4) Bonus Sections: The analysis includes an exploration of advanced techniques such as feature selection and class balancing using methods like SMOTE and NearMiss to further improve model performance and address data imbalances.

## Repository Contents

- `notebooks/`: Contains the Jupyter Notebooks for Exploratory Data Analysis (EDA) and model evaluation.

- `src/`: Holds custom Python classes and functions, including ClassifierCV and ClassSpecificImputer.

- `models/`: Stores the final trained model (.pkl file).

- `figures/`: Contains all generated plots and visualizations from the analysis.

- `requirements.txt`: A list of all necessary Python dependencies.

## Getting Started

To run the project, follow these steps to set up your environment:

- Clone the Repository:

```Bash
git clone https://github.com/dvoulgari/Breast-Cancer-Diagnosis-ML.git
cd Breast-Cancer-Diagnosis-ML
```
- Install Dependencies: Install the required Python libraries.

```Bash
pip install -r requirements.txt
```
- Launch Jupyter Notebook: Open the main notebook to view the code and results.

```Bash
jupyter notebook
```
