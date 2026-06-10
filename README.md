# Heart Disease Prediction System

> A production-ready web application leveraging Machine Learning through Xgboost with 87.04% accuracy to assess cardiovascular risk profiles. The system features a multi-staged preprocessing pipeline, custom cluster feature injection, and SHAP-driven explainable AI diagnostics to deliver interpretable risk scores.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Machine Learning Pipeline](#architecture--machine-learning-pipeline)
- [Database Schema & Security](#database-schema--security)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Feature Engineering & Parameters](#feature-engineering--parameters)
- [Explainable AI (XAI) & Diagnostics](#explainable-ai-xai--diagnostics)
- [Future Enhancements](#future-enhancements)

---

## Overview

This project implements an end-to-end medical decision support web application that assesses a patient's risk of cardiovascular disease based on clinical and behavioral attributes.

Rather than relying on raw categorical mapping, the system introduces a novel **hybrid preprocessing pipeline**. It segments patient attributes using gender-specific **K-Modes clustering** models (`km_male.pkl` and `km_female.pkl`) and appends the resulting sub-population cluster index as an active feature. This contextual identifier, combined with engineered health indices, feeds into a tuned **XGBoost Classifier** (`xgb_model.pkl`) to predict overall risk and exact statistical probabilities.

To bridge the gap between complex black-box modeling and clinical utility, the system utilizes **SHAP (SHapley Additive exPlanations)** to compute localized feature attributions in real time, generating personalized, data-driven preventative health recommendations.

---

## Architecture & Machine Learning Pipeline

```
User Input (Clinical & Behavioral Metrics)
        │
        ├──► Feature Engineering (Compute BMI & Mean Arterial Pressure)
        ├──► Discretization & Binning (Age, BMI, MAP categories)
        │
  ┌─────┴────────────────────────────────────────┐
  │ Cluster Assignment (Gender-Specific K-Modes) │
  │  ├─ Gender = 2 ──► km_male.pkl               │
  │  └─ Gender = 1 ──► km_female.pkl             │
  └─────┬────────────────────────────────────────┘
        │
  Unified Feature Vector (Including Cluster Assignment)
        │
  ┌─────┴────────────────────────────────────────┐
  │ Inference Engine                             │
  │  ├─ XGBoost Classifier ──► Risk Prediction   │
  │  └─ SHAP TreeExplainer ──► Feature Impact    │
  └─────┬────────────────────────────────────────┘
        │
  Output Interface: Binary Risk State, Probability %,
  Dynamic ChartJS Visualization, & Targeted Clinical Recs
```

### Class-Conditioned Logic Flow

1. **Dynamic Feature Expansion:** Raw metrics like height, weight, systolic (`ap_hi`), and diastolic (`ap_lo`) pressure are immediately processed into composite indicators (BMI and MAP).
2. **Contextual Clustering:** The application routes the processed attributes to dedicated clustering models based on biological sex to resolve hidden epidemiological sub-groups.
3. **Risk & Impact Extraction:** The fully engineered vector passes into the XGBoost architecture while simultaneously triggering a local `TreeExplainer` sweep to isolate exactly which metrics drove the prediction upward.

---

## Database Schema & Security

The application utilizes an isolated relational database structure managed via `Flask-SQLAlchemy` to maintain secure user state and access controls.

### Authentication Database Structure

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | Primary Key, Auto-Increment | Unique internal identifier for the user. |
| username | VARCHAR(150) | Unique, Nullable=False | Account authentication identifier. |
| password | VARCHAR(150) | Nullable=False | Salted and hashed credential storage. |

### Security Implementations

- **Cryptographic Hashing:** Credentials are never written to disk in plain text. Registration routes process input via `werkzeug.security` utilizing the **PBKDF2 with SHA-256** iterative derivation algorithm.
- **Session Management:** Component routes are secured using a `Flask-Login` session manager, strictly restricting the predictive interface to validated accounts via the `@login_required` middleware stack.
- **Instance Isolation:** Relational SQLite tracking layers are constrained to a decoupled internal `/instance` directory configuration to block directory-traversal vulnerabilities.

---

## Features

- **Advanced Feature Pipeline:** Automatically engineers clinical standard metrics including Body Mass Index (BMI) and Mean Arterial Pressure (MAP) from raw physical inputs.
- **Dual K-Modes Clustering Integration:** Incorporates specialized unsupervised learning vectors directly into a supervised classification task to isolate sex-stratified medical tendencies.
- **Explainable AI (XAI):** Executes localized mathematical feature breakdowns via SHAP tree-explainers on every individual prediction, preventing arbitrary decision generation.
- **Deterministic Recommendation System:** Maps localized high-impact positive SHAP values directly to an actionable, tailored lifestyle modification rule engine.
- **Interactive Data Visualization:** Serializes localized feature weight arrays into JSON formats to feed clean, front-end canvas components (Chart.js) for immediate visual analysis.
- **Secure Enterprise Wrapper:** Built-in user registration, session tracking, cryptographic validation, and automated platform directory configuration.

---

## Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Web Architecture | Flask, Flask-SQLAlchemy, Flask-Login, Jinja2 Templates |
| Machine Learning | XGBoost, KModes (scikit-learn compatible clusterers) |
| Explainable AI | SHAP (SHapley Additive exPlanations) |
| Data Processing | Pandas, NumPy |
| Security Engine | Werkzeug Cryptography Suite |
| Front-End Stack | HTML5, CSS3 Grid/Flexbox, Chart.js (JSON-driven) |
| Persistence Layer | SQLite3 |

---

## Project Structure

```
heart-disease-prediction/
│
├── app.py                          # Application core: Routing, authentication, and pipelines
│
├── instance/                       # Local database runtime storage
│   └── users.db                    # Auto-generated SQLite credentials repository
│
├── models/                         # Serialized predictive binary artifacts
│   ├── xgb_model.pkl               # Optimized XGBoost classification model
│   ├── km_male.pkl                 # K-Modes clusterer for male sub-cohort analysis
│   └── km_female.pkl               # K-Modes clusterer for female sub-cohort analysis
│
├── templates/                      # Front-end structural rendering templates
│   ├── login.html                  # User authentication interface
│   ├── register.html               # Account provisioning template
│   └── index.html                  # Core diagnostics and data dashboard
│
└── README.md                       # Architectural and deployment documentation
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- C++ Build Tools (Required for compiling local dependencies like SHAP)
- Virtual Environment Manager (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Configure Environment & Dependencies

Initialize an isolated runtime wrapper and run installations using the Python package manager:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

pip install flask flask-sqlalchemy flask-login
pip install pandas numpy xgboost shap kmodes
```

### 3. Deploy Serialized Artifacts

Ensure your trained binary files are placed inside the `/models` directory exactly as structured below:

- `models/xgb_model.pkl`
- `models/km_male.pkl`
- `models/km_female.pkl`

---

## Usage

### Initializing the Application Runtime

Execute the main script from your terminal:

```bash
python app.py
```

The system will verify local folder parameters, build the SQLite instance schema dynamically if absent, and host the web server locally at `http://127.0.0.1:5000/`.

### Administrative Python Script: Model Inference Testing

To evaluate your local model integration pipeline independent of the browser wrapper, execute this isolated inference block:

```python
import pickle
import pandas as pd
import numpy as np

# Load the core model bundle
with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Mock input representation array
# Features: [gender, cholesterol, smoke, alco, gluc, active, BMI, map, age, cluster]
mock_patient_record = pd.DataFrame(
    [[1, 3, 0, 0, 1, 1, 4, 3, 5, 1]],
    columns=['gender', 'cholesterol', 'smoke', 'alco', 'gluc', 'active', 'BMI', 'map', 'age', 'cluster']
)

# Extract predictions
prediction = model.predict(mock_patient_record)[0]
probability = model.predict_proba(mock_patient_record)[:, 1][0]

print(f"Risk Classification Result: {prediction} (Target Indicator)")
print(f"Calculated Core Risk Probability: {probability * 100:.2f}%")
```

---

## Feature Engineering & Parameters

Input vectors are subjected to a multi-stage deterministic categorization workflow before scoring.

### Mathematical Formulas

**Body Mass Index (BMI):**

$$\text{BMI} = \frac{\text{Weight (kg)}}{\left(\frac{\text{Height (cm)}}{100}\right)^2}$$

**Mean Arterial Pressure (MAP):**

$$\text{MAP} = \frac{(2 \times \text{Diastolic BP}) + \text{Systolic BP}}{3}$$

### Discretization Boundaries

The engineered features are mapped into integer bins before entering the models:

| Target Metric | Bin Assignments |
|---------------|-----------------|
| **Age** | `[30-35): 0`, `[35-40): 1`, `[40-45): 2`, `[45-50): 3`, `[50-55): 4`, `[55-60): 5`, `[60-65): 6`, `[65+): 7` |
| **BMI** | `(<18.5): 0`, `[18.5-25): 1`, `[25-30): 2`, `[30-35): 3`, `[35-40): 4`, `(40+): 5` |
| **MAP** | `(<70): 0`, `[70-80): 1`, `[80-90): 2`, `[90-100): 3`, `[100-110): 4`, `(110+): 5` |

---

## Explainable AI (XAI) & Diagnostics

The integration of SHAP ensures that clinicians and end-users can visually interpret *why* an evaluation skewed high or low.

```python
# Real-time extraction logic executed within the application core
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(processed_data)
```

### Diagnostic Output Map

- **Positive Attribution Weights (+x):** Represents indicators that actively pulled the classification score towards a high-risk diagnosis.
- **Negative Attribution Weights (-x):** Indicates protective habits or healthy metrics that drop the cumulative risk baseline.
- **Visual Serialization:** Feature outputs for gender and administrative cluster groups are systematically excluded from the output dictionary. The remaining indicators are sorted by absolute magnitude, converted to standard float arrays, and pushed to front-end visualization matrices to chart specific patient exposure risks.



