# LiverGuard — Liver Disease Risk Prediction (Vercel Web App)

A clean, white-theme web app that predicts liver-disease risk from 10 biosensor
readings using an ensemble ML model trained on the **Indian Liver Patient Dataset
(Kaggle, 583 records)**. Serverless-ready — deploys to Vercel with one click.

## ✨ Features
- **White / light theme** with soft gradients and cards
- Enter sensor readings → instant risk %, risk level, and out-of-range flags
- Model performance dashboard (accuracy, AUC, recall, F1)
- IoT sensor integration map (which sensor measures which parameter)
- `POST /api/predict` JSON API for ESP32 / microcontroller integration

## 🧠 Model
Soft-voting ensemble (Random Forest + Gradient Boosting + SVM + Logistic Regression),
sklearn-only for a small, fast serverless function.

| Metric | Value |
|---|---|
| Accuracy | 74.0% |
| ROC-AUC | 0.816 |
| Recall | 91.3% |
| F1 Score | 83.3% |
| Precision | 76.6% |

High recall means the device catches almost every true patient — the key
requirement for a screening tool.

## 🚀 Deploy to Vercel (one click)
1. Push this folder to a GitHub repo (already structured for Vercel).
2. Go to [vercel.com](https://vercel.com) → **Add New → Project** → import the repo.
3. Vercel auto-detects the Python build. Click **Deploy**.
4. Done — you get a public URL.

The key files:
- `vercel.json` — routes every path to `api/index.py` (the Flask app)
- `api/index.py` — Flask app + model loading + UI
- `api/liver_model.pkl` — the trained model
- `api/model_metrics.json` — metrics shown in the dashboard
- `requirements.txt` — pinned dependencies

## 🔌 API (for your IoT device)
```json
POST /api/predict
{
  "age": 52, "gender": "male",
  "total_bilirubin": 3.9, "direct_bilirubin": 2.0,
  "alkaline_phosphotase": 195, "alt": 27, "ast": 59,
  "total_proteins": 7.3, "albumin": 2.4, "ag_ratio": 0.49
}
→ { "risk_probability": 88.4, "prediction": 1, "risk_level": "HIGH" }
```
ESP32 (Arduino): use `HTTPClient` to POST the JSON, then drive an LED / buzzer /
OLED from the `risk_level` field.

## 🖥️ Run locally
```bash
pip install -r requirements.txt
python run_local.py   # serves on http://localhost:8010
```

> Academic screening prototype — not a medical device.
