# 🫀 Heart Disease Prediction — ML Classification with GridSearch Optimization

> **Predicting cardiovascular disease risk from clinical patient data using an ensemble of optimized machine learning classifiers — combining unsupervised clustering with supervised learning for richer, gender-aware risk profiling.**

---

## 📌 Key Features

- **Multi-Model Benchmarking** — Trains and evaluates three distinct classifiers (XGBoost, Random Forest, K-Nearest Neighbors) side-by-side, giving a full picture of the accuracy–interpretability tradeoff.
- **GridSearchCV Hyperparameter Tuning** — Applies exhaustive 5-fold cross-validated grid search across all three models to discover the optimal hyperparameter set, eliminating guesswork and preventing over-fitting.
- **Gender-Stratified KModes Clustering** — Employs unsupervised categorical clustering (KModes with Huang initialization) separately for male and female cohorts, injecting a learned risk-cluster feature into the supervised pipeline.
- **Clinical Feature Engineering** — Derives medically meaningful features including Body Mass Index (BMI), Mean Arterial Pressure (MAP), and age bands; then bins all three into clinically standard risk categories.
- **Robust IQR-Based Outlier Removal** — Detects and eliminates physiologically implausible readings in blood pressure, height, and weight before any model sees the data, directly improving generalization.
- **Comprehensive EDA Suite** — Generates correlation heatmaps, categorical count plots with percentage labels, and per-column box plots to surface class imbalance and feature relationships before modeling.
- **Stratified Train/Test Split** — Preserves the target class ratio across train and test sets, ensuring evaluation metrics are not distorted by imbalance.
- **Full Classification Reporting** — Outputs confusion matrices (visualized as annotated heatmaps) and complete `classification_report` tables (precision, recall, F1) for every model.

---

## 🛠️ Tech Stack & Tools

**Language**
`Python 3.x`

**Data & ML**
| Library | Role |
|---|---|
| `pandas` | Data loading, wrangling, feature engineering |
| `numpy` | Numerical operations, quantile-based outlier bounds |
| `scikit-learn` | Train/test split, GridSearchCV, RF, KNN, metrics |
| `xgboost` | Gradient-boosted tree classifier |
| `kmodes` | Categorical clustering (KModes algorithm) |

**Visualization**
| Library | Role |
|---|---|
| `matplotlib` | Base plotting layer |
| `seaborn` | Heatmaps, count plots, box plots |

**Environment**
`Jupyter Notebook` · `cardio_train.csv` (Kaggle Cardiovascular Disease Dataset)

---

## 🏗️ Architecture & Data Flow

```
Raw CSV (cardio_train.csv)
        │
        ▼
┌───────────────────┐
│  Exploratory      │  → Correlation heatmap, count plots,
│  Data Analysis    │    box plots, duplicate/null checks
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  → Drop ID column
│                   │  → IQR outlier removal (ap_hi, ap_lo,
│                   │    height, weight at 2.5–97.5 percentile)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Feature          │  → Compute BMI, MAP
│  Engineering      │  → Bin age → 7 risk bands
│                   │  → Bin BMI → 6 WHO categories
│                   │  → Bin MAP → 6 pressure categories
│                   │  → Drop raw source columns
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  KModes           │  → Split by gender (male / female)
│  Clustering       │  → Fit k=2 clusters per cohort (Huang init)
│                   │  → Re-merge with cluster label as new feature
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stratified       │  → 80% train / 20% test
│  Train/Test Split │    stratified on target `cardio`
└───────────────────┘
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   XGBoost         Random          KNN
  (+ GridSearch)   Forest          (+ GridSearch)
                  (+ GridSearch)
        │              │              │
        └──────────────┴──────────────┘
                        │
                        ▼
          Confusion Matrix · Accuracy
          Precision · Recall · F1-Score
```

Each model is first trained with default parameters as a baseline, then re-trained using the best hyperparameters discovered by `GridSearchCV` (5-fold CV, `accuracy` scoring), allowing direct measurement of the optimization gain.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kmodes jupyter
```

Or if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

### 4. Obtain the dataset
Download `cardio_train.csv` from the [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and place it in the project root directory.

### 5. Launch Jupyter and run the notebook
```bash
jupyter notebook Heartdisease_with_GridSearch.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).

---

## 🗺️ Future Roadmap

1. **SHAP Explainability Layer** — Integrate SHAP (SHapley Additive exPlanations) to generate per-prediction feature importance plots, making the model interpretable enough for a clinical audience and satisfying explainability requirements in healthcare AI.

2. **SMOTE / Class-Balancing Pipeline** — Introduce Synthetic Minority Over-sampling Technique (SMOTE) inside a proper `Pipeline` object to handle any residual class imbalance without leaking resampled data into the validation fold — a common but critical pitfall.

3. **Streamlit Deployment** — Wrap the best-performing tuned model in a lightweight Streamlit web app, allowing a clinician (or recruiter 😄) to input patient vitals and receive an instant cardiovascular risk prediction with confidence score.
