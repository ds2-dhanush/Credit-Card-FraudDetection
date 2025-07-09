# 💳 Credit Card Fraud Detection – End‑to‑End Project

This repository contains a production‑ready machine‑learning pipeline that detects fraudulent credit‑card transactions and exposes the model through an interactive Streamlit web application.

---
## 📌 Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Architecture](#project-architecture)
4. [Approach](#approach)
5. [Model Performance](#model-performance)
6. [Web Application Features](#web-application-features)
7. [How to Run Locally](#how-to-run-locally)
8. [How It Works Under the Hood](#how-it-works-under-the-hood)
9. [Why It Is Useful](#why-it-is-useful)
10. [Deployment Guide](#deployment-guide)
11. [Future Work](#future-work)
12. [Author](#author)

---
## Problem Statement
Financial institutions lose billions to credit‑card fraud every year.  
The goal is to build a model that flags suspicious transactions **before** money leaves the bank, while minimising false alarms that annoy genuine customers.

---
## Dataset
| Column | Description |
|--------|-------------|
| trans_date_trans_time | Timestamp of the transaction |
| cc_num | Credit‑card number (anonymised) |
| merchant / category | Where & what was purchased |
| amt | Transaction amount |
| street • city • state | Billing location |
| lat • long | Card‑holder location |
| merch_lat • merch_log | Merchant location |
| distance | Engineered Haversine distance (km) |
| hour • day • weekday • month | Engineered calendar features |
| is_fraud | **Target** (1 = fraud, 0 = legit) |

The raw data were strongly imbalanced (≈ 0.3 % fraud). We applied **SMOTE** to oversample the minority class during training.

---
## Project Architecture
```
data/           ← raw & processed CSVs
notebooks/      ← EDA & modelling notebooks
models/
  └─ fraud_detector_model.pkl
app.py          ← Streamlit front‑end
requirements.txt
```

---
## Approach
1. **Exploratory Data Analysis**: identified distribution skews, outliers, temporal & spatial fraud patterns.  
2. **Feature Engineering**  
   • Extracted hour / weekday / month  
   • Calculated geo‑distance between cardholder and merchant  
   • Label‑encoded high‑cardinality categories  
3. **Class‑Imbalance Handling**: `SMOTE` on training split.  
4. **Model Benchmarking**: Logistic Regression ↔ Random Forest ↔ XGBoost.  
5. **Chosen Model**: **XGBoost** (best ROC‑AUC 0.95, fraud‑recall 90 %).  
6. **Web Application**: Streamlit app offering CSV upload **and** manual form entry with confidence scores.

---
## Model Performance
| Metric (Test Set) | Logistic Regression | Random Forest | **XGBoost (Final)** |
|-------------------|---------------------|---------------|---------------------|
| Accuracy          | 94 %               | 99 %          | 99 %                |
| Recall (Fraud)    | 73 %               | 72 %          | **90 %**            |
| Precision (Fraud) | 4 %                | **51 %**      | 38 %                |
| ROC‑AUC           | 0.83               | 0.86          | **0.95**            |

> **Interpretation**: XGBoost detects 9 out of 10 frauds with an acceptable false‑alarm rate, outperforming classical baselines.

---
## Web Application Features
- **CSV Upload**: bulk‑scoring of thousands of transactions; downloadable results.  
- **Manual Form**: real‑time single‑transaction verdict; dropdowns with common Indian merchants, plus custom entry.  
- **Confidence Score**: displays predicted fraud probability.  
- **Clean UI**: built with Streamlit, responsive on mobile.  

![screenshot](projects/fraud-app.png)

---
## How to Run Locally
```bash
cd fraud-detector
pip install -r requirements.txt
streamlit run app.py
```

---
## How It Works Under the Hood
1. **Pre‑processing pipeline** loads the 15 core features, applies scaling & encoding.  
2. **XGBoost** produces a fraud probability ∈ [0, 1].  
3. Threshold = 0.5 → binary class.  
4. Streamlit front‑end displays label & confidence, colour‑coded (green ℹ/ red ⚠).

The model pays most attention to **amount**, **distance**, **transaction hour**, and unusual **merchant–location pairs**.

---
## Why It Is Useful
- **Banks**: proactive fraud blocking saves charge‑back costs.  
- **Support Teams**: manually investigate high‑confidence alerts.  
- **Customers**: fewer false declines thanks to balanced precision–recall.  
- **Recruiters / Interviewers**: demonstrates full ML life‑cycle skills (EDA → modelling → deployment).

---
## Deployment Guide
- **Model & app** hosted free on **Streamlit Cloud**  
  `https://fraud-detector.streamlit.app`  
- **Portfolio site** on **GitHub Pages** linking to the app & source code.

---
## Future Work
- Hyper‑parameter tuning (Randomised Search) for higher precision.  
- Real‑time streaming via Kafka + REST API.  
- Explainable AI layer (SHAP) to visualise feature importance per prediction.

---
## Author
**Dhanush R**  
- Portfolio: https://ds2-dhanush.github.io/My-Portfolio/ 
- LinkedIn: https://www.linkedin.com/in/dhanush-r-a12007287  
- GitHub:   https://github.com/ds2-dhanush?tab=repositories