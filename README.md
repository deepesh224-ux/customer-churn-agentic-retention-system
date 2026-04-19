# Customer Churn Prediction and Agentic Retention Strategy System

## Project Overview
This repository contains an end-to-end artificial intelligence system designed to identify at-risk customers and generate structured, data-driven retention strategies. The project is divided into two primary phases: a machine learning pipeline for churn classification and an agentic AI assistant for personalized intervention planning.

## Milestone 1: Data Preprocessing & Pipeline
The current phase focuses on building a robust automated data pipeline that transforms raw customer data into a machine-learning-ready format.

### Key Features
* **Automated Ingestion:** Uses `kagglehub` to fetch the latest Telco Churn dataset directly.
* **Data Cleaning:** Handles missing values in `TotalCharges` and removes non-predictive features like `customerID`.
* **Feature Engineering:** Implements Binary Encoding and One-Hot Encoding for categorical variables.
* **High-Performance Export:** Saves processed data in **Parquet** format to preserve data types and optimize speed.

## Milestone 2: Model Training, Evaluation & Retention Placeholder

The second phase introduces a formal training pipeline, robust evaluation, and the foundation for agentic retention strategies.

### Key Features
* **Dedicated Training Pipeline (`src/train.py`):** A standalone script that handles data loading, preprocessing, model training, and saving artifacts.
* **Robust Evaluation:** Implements 5-fold cross-validation to ensure model stability and reports Accuracy, ROC-AUC, and Confusion Matrix.
* **Retention Automation Layer (`src/retention_automation.py`):** A modular placeholder for the upcoming agentic strategy engine, including placeholder logic for personalized outreach.
* **Comprehensive Testing:** Unit tests for both preprocessing cleaning steps and the retention module.

---

##  Setup & Installation

Follow these steps to replicate the environment on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/Vegapunk-debug/customer-churn-agentic-retention-system.git](https://github.com/Vegapunk-debug/customer-churn-agentic-retention-system.git)
cd customer-churn-agentic-retention-system
```
### 2. Create and Activate Virtual Environment
```bash
# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

1. **Preprocessing:** Use the `notebooks/preprocessing.ipynb` to download and prepare the initial dataset.
2. **Training:** Run the training script to generate the model and evaluation metrics:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 src/train.py
   ```
3. **Inference / App:** Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Unit Testing

To ensure data integrity and reliability, run the unit tests:
```bash
python3 -m unittest discover tests
```

## Evaluation Results

After running `src/train.py`, metrics are saved to `models/evaluation_results.json`. 

| Metric | Value (Baseline) |
| :--- | :--- |
| **Accuracy** | ~80% |
| **ROC-AUC** | ~84% |
| **Cross-Validation Accuracy** | ~79% (+/- 1%) |

## Future Roadmap: Agentic Retention
The `src/retention_automation.py` module is ready for integration. Future updates will include:
- [ ] LLM-powered personalized retention message generation.
- [ ] Automated discount/coupon assignment based on churn risk.
- [ ] Integration with CRM tools for automated outreach.


