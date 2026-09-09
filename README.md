<div align="center">

# 📉 Telco Customer Churn — Prediction & Analysis

**Predicting which telecom customers are about to leave — and why.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

Customer churn — when a customer cancels their subscription — is one of the costliest problems for a telecom company, since acquiring a new customer is far more expensive than retaining an existing one.

This project builds an end-to-end churn-prediction pipeline on the **IBM Telco Customer Churn** dataset (7,043 customers, 21 attributes) and deploys the best model as a **FastAPI REST API** with a **plain HTML/CSS/JS frontend**.

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── app/
│   └── main.py                          # FastAPI application
│
├── model/                               # Generated model artifacts (after running export script)
│   ├── adaboost_model.joblib
│   ├── onehot_encoder.joblib
│   └── feature_columns.json
│
├── frontend/                            # Static frontend (deploy to Vercel)
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── vercel.json
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── images/                              # EDA charts from notebook
│
├── Telco_Customer_Churn_Prediction.ipynb # Original analysis notebook
├── train_and_export.py                  # One-time: retrain & save model
├── Procfile                             # Render / Railway start command
├── requirements.txt                     # Python dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train & export the model (one-time)

```bash
python train_and_export.py
```

This will create the `model/` directory with the serialized model and encoder.

### 4. Run the FastAPI API locally

```bash
uvicorn app.main:app --reload
```

The API will be available at **http://localhost:8000**.

- Swagger UI docs: http://localhost:8000/docs
- Health check: http://localhost:8000/

### 5. Test the `/predict` endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Example response:**

```json
{
  "prediction": "Yes",
  "churn_probability": 0.5765
}
```

### 6. Run the frontend locally

Option A — Python (no extra installs):

```bash
cd frontend
python -m http.server 5500
```

Option B — VS Code Live Server extension on port 5500.

Then open **http://localhost:5500** in your browser.

> The frontend is pre-configured to call `http://localhost:8000`. Both the API and frontend must be running at the same time.

---

## 🌐 Deployment

### Deploy the API to Render

1. Push the repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service**.
3. Connect your GitHub repository.
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:**
     - `ALLOWED_ORIGINS` = your Vercel frontend URL (e.g. `https://your-app.vercel.app`)
5. Deploy. Copy the Render URL (e.g. `https://your-api.onrender.com`).

### Deploy the API to Railway (alternative)

1. Push the repo to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → Deploy from GitHub.
3. Railway will auto-detect the `Procfile`.
4. Add environment variable: `ALLOWED_ORIGINS` = your Vercel frontend URL.
5. Deploy. Copy the Railway URL.

### Deploy the frontend to Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**.
2. Import your GitHub repository.
3. Set **Root Directory** to `frontend`.
4. Set **Framework Preset** to `Other`.
5. Leave the build command **empty** (no build step needed).
6. Set **Output Directory** to `.` (current directory).
7. Deploy.

### Update the API URL

After deploying the API, update the `API_URL` in [`frontend/script.js`](frontend/script.js):

```javascript
// Change this line:
const API_URL = "http://localhost:8000";

// To your deployed API URL:
const API_URL = "https://your-api.onrender.com";
```

---

## 📊 Model Details

| Item | Detail |
|---|---|
| **Algorithm** | AdaBoost (best F1 on churn class) |
| **Dataset** | IBM Telco Customer Churn (7,043 rows) |
| **Features** | 22 (after encoding) |
| **Target** | Churn (binary) |
| **Class balancing** | SMOTE on training set |
| **Tuning** | RandomizedSearchCV (F1, 3-fold CV) |
| **Test F1 (Churn)** | ~0.61 |
| **Test ROC-AUC** | ~0.83 |

### Input Features

| Feature | Type | Values |
|---|---|---|
| SeniorCitizen | int | 0 or 1 |
| Partner | str | Yes, No |
| Dependents | str | Yes, No |
| tenure | int | 0–72 |
| MultipleLines | str | Yes, No, No phone service |
| InternetService | str | DSL, Fiber optic, No |
| OnlineSecurity | str | Yes, No, No internet service |
| OnlineBackup | str | Yes, No, No internet service |
| DeviceProtection | str | Yes, No, No internet service |
| TechSupport | str | Yes, No, No internet service |
| StreamingTV | str | Yes, No, No internet service |
| StreamingMovies | str | Yes, No, No internet service |
| Contract | str | Month-to-month, One year, Two year |
| PaperlessBilling | str | Yes, No |
| PaymentMethod | str | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| MonthlyCharges | float | e.g. 29.85 |
| TotalCharges | float | e.g. 29.85 |

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` · `FastAPI` · `Uvicorn` · `HTML/CSS/JS`

---

## 📁 Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 rows × 21 columns.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
