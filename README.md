# 🚀 Startup Success Prediction Model

## 📌 Project Overview

This project is an end-to-end Machine Learning pipeline designed to predict the success or failure of a startup based on various business and financial metrics. The model utilizes **Logistic Regression** and is specifically engineered to handle highly imbalanced real-world venture capital datasets.

## 📊 Dataset Features

The model expects a dataset (`startup_success_dataset.csv`) with the following core features:

- **Numeric:** `funding_rounds`, `founder_experience_years`, `team_size`, `market_size_billion`, `product_traction_users`, `burn_rate_million`, `revenue_million`.
- **Categorical:** `investor_type`, `sector`, `founder_background`.
- **Target:** `outcome` (Automatically mapped to a binary `is_success` target).

## 🛠️ Technical Highlights (Under the Hood)

This system is built with Enterprise-level Machine Learning standards:

- **Robust Imputation:** Uses `median` for numeric columns to resist extreme financial outliers, and `constant` for categorical data to retain missing value signals.
- **Dummy Variable Trap Prevention:** Implements `drop_first=True` during One-Hot Encoding to avoid perfect multicollinearity.
- **Data Leakage Prevention:** Strict isolation of `fit_transform` on the Training set, applying only `transform` on Validation and Test sets.
- **Imbalanced Data Handling:** Applies `class_weight='balanced'` to penalize the model heavily for missing rare successful startups (Unicorns).
- **Comprehensive Visualization:** Generates a 3-part dashboard including a Confusion Matrix, ROC Curve (with AUC), and Feature Importance weights.

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Installation

```bash
python -m venv env
```

````bash
source env/bin/activate
```

```bash
pip install -r requirements.txt
````

```bash
python main.py
```

HAPPY CODING !!
