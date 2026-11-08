# Breast Tumour Classification & ML Analysis Dashboard

An interactive machine learning project that explores breast tumour classification using supervised learning techniques. The project compares Logistic Regression and Random Forest classifiers and presents the results through an interactive Streamlit dashboard.

**Built using:** Python · Scikit-learn · Pandas · NumPy · Streamlit · Plotly

---

## 1. Project Overview

Breast cancer classification is a binary classification problem in which tumour observations are categorised as either malignant or benign based on numerical measurements of cell nuclei.

This project aims to understand and implement a complete machine learning workflow, including data exploration, preprocessing, model training, performance evaluation and interactive visualisation.

Two supervised machine learning algorithms are implemented and compared:

- Logistic Regression
- Random Forest Classifier

A Streamlit-based dashboard allows users to explore the dataset, examine model performance and experiment with predictions using manually entered feature values or existing test samples.

> **Disclaimer:** This project is developed strictly for educational and learning purposes. It has not been clinically or scientifically validated and must not be used for medical diagnosis, treatment decisions or health assessments. Predictions should not be considered medically reliable.

---

## 2. Dataset Description

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available through Scikit-learn.

The dataset contains measurements computed from digitised images of fine needle aspirates (FNA) of breast masses.

| Dataset Attribute | Details |
|---|---|
| Total observations | 569 |
| Number of features | 30 |
| Target variable | Diagnosis |
| Classification type | Binary classification |
| Missing values | None |
| Data source | Scikit-learn |

### Target Classes

The target variable contains two categories:

- **Malignant (0):** Indicates a malignant tumour.
- **Benign (1):** Indicates a benign tumour.

### Feature Categories

The 30 numerical features are divided into three groups:

1. Mean features – Average measurements of cell nuclei.
2. Standard Error features – Variability in the measurements.
3. Worst features – Mean of the three largest values for each measurement.

These include characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset's structure, distribution and relationships between variables.

The analysis included:

- Examining dataset dimensions and feature information.
- Checking for missing values and duplicate records.
- Analysing the distribution of malignant and benign observations.
- Visualising feature distributions.
- Studying relationships between numerical features.
- Generating a correlation heatmap to identify highly correlated variables.

The dataset contains approximately:

- 62.7% Benign observations.
- 37.3% Malignant observations.

---

## 4. Data Preprocessing

The following preprocessing steps were implemented:

- Separation of independent features and the target variable.
- Train-test split using an 80:20 ratio.
- Stratified sampling to preserve class proportions across training and testing sets.
- Feature standardisation using `StandardScaler` for Logistic Regression.

The same training and testing split was used to compare both classifiers.

---

## 5. Machine Learning Models

### A. Logistic Regression

Logistic Regression was implemented as the primary classification model.

A Scikit-learn pipeline was created combining:

- StandardScaler
- LogisticRegression

The model estimates class probabilities using the sigmoid function and assigns a binary classification based on the predicted probability.

### B. Random Forest Classifier

Random Forest was implemented as a comparison model.

It uses an ensemble of decision trees to generate predictions and was configured with 300 estimators.

Both models were trained and evaluated using the same dataset split.

---

## 6. Model Evaluation

Model performance was evaluated using multiple classification metrics rather than relying exclusively on accuracy.

The evaluation included:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Stratified 5-Fold Cross-Validation

### Test Set Performance

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 98.25% | 94.74% |
| Malignant Recall | 97.62% | 92.86% |
| Test Samples | 114 | 114 |

### Cross-Validation Results

Stratified 5-fold cross-validation was used to evaluate model performance across multiple data partitions.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Mean Accuracy | 97.37% | 95.26% |
| Accuracy Standard Deviation | 1.66% | 1.31% |
| Macro F1-score | 97.14% | 94.89% |
| Malignant Recall | 94.36% | 92.95% |

*Cross-validation values represent the mean scores across five folds.*

### ROC-AUC

Logistic Regression achieved a ROC-AUC score of approximately **0.9954** on the held-out test set when malignant observations were treated as the positive class.

### Model Selection

Logistic Regression was selected as the primary model for the dashboard because it achieved better results than Random Forest across the reported test-set and cross-validation evaluations.

These results are specific to the dataset and experimental setup and do not establish that Logistic Regression is universally superior.

---

## 7. Feature Analysis

Feature coefficients from the trained Logistic Regression model were examined to understand how individual standardised features influence the model's predictions.

The dashboard displays the features with the largest absolute coefficient values.

Coefficient direction indicates the relationship with the predicted class, holding other features constant.

Since several dataset features are correlated, coefficient magnitudes should be interpreted cautiously. They do not establish medical causation or clinical significance.

---

## 8. Streamlit Dashboard Features

The project includes an interactive dashboard with the following sections:

### 1. Overview
- Introduction to the project.
- Dataset information.
- Model overview.
- Educational-use disclaimer.

### 2. Predict Classification
- Manual entry of the 30 numerical features.
- Selection of existing test samples.
- Logistic Regression-based predictions.
- Display of predicted class probabilities.

### 3. Dataset Analytics
- Malignant and benign class distribution.
- Feature correlation heatmap.
- Feature distribution visualisations.

### 4. Feature Insights
- Top Logistic Regression coefficients.
- Feature contribution direction.
- Coefficient analysis table.

### 5. Model Performance
- Logistic Regression vs Random Forest comparison.
- Confusion matrices.
- Classification reports.
- Cross-validation performance.
- Evaluation metrics.

---

## 9. Project Structure

breast-tumour-classification/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   └── mediclass_model.pkl
│
├── src/
│   ├── dataset.py
│   ├── dataexploration.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_comparison.py
│   ├── random_forest.py
│   └── feature_analysis.py
│
├── notebooks/
│   └── data_exploration.ipynb
│
└── screenshots/
    ├── 01_confusion_matrix.png
    ├── 02_dashboard.png
    ├── 03_feature_correlations.png
    ├── 04_predict.png
    
---
## 10. Installation and Execution

### Prerequisites

- Python 3.9 or later
- pip package manager
- Git

### Step 1: Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/sar-isha07/Breast-Tumour-Classification.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Breast-Tumour-Classification
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser. If it doesn't, open the local URL displayed in your terminal.

### Optional: Retrain the Model

To train the models and regenerate the saved Logistic Regression model, run:

```bash
python train_model.py
```

After successful execution, launch the application using the command in Step 4.

## 11. Limitations

- The dataset contains only 569 observations and may not represent the diversity of real-world clinical populations.
- The models are trained and evaluated on a single publicly available dataset.
- Performance metrics may vary with different train-test splits and datasets.
- Logistic Regression coefficients do not establish causality or medical significance.
- The application has not undergone clinical validation or independent medical evaluation.

The project is intended to demonstrate machine learning concepts and should not be interpreted as a medical diagnostic system.

---

## 12. Technologies Used

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Streamlit | Interactive web application |
| Plotly | Interactive data visualisation |
| Matplotlib & Seaborn | Exploratory data visualisation |
| Joblib | Model serialisation and loading |
| Jupyter Notebook | Data exploration and experimentation |

---

## 13. Learning Outcomes

Through this project, the following concepts were explored and implemented:

- Exploratory Data Analysis and data visualisation.
- Binary classification using supervised learning.
- Logistic Regression and Random Forest.
- Feature standardisation and preprocessing pipelines.
- Train-test splitting and stratified sampling.
- Cross-validation and model comparison.
- Classification metrics and ROC-AUC.
- Feature coefficient interpretation.
- Deployment of a machine learning model using Streamlit.

---

**Project Category:** Machine Learning | Binary Classification | Data Science

**Purpose:** Educational and academic project.
