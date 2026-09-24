"""
LiverGuard — Liver Disease Risk Prediction (Vercel-ready Flask app)
Lean sklearn-only ensemble model for fast serverless cold-starts.
"""
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
_bundle = joblib.load(os.path.join(HERE, "liver_model.pkl"))
MODEL = _bundle["model"]
FEATURES = _bundle["feature_columns"]
try:
    with open(os.path.join(HERE, "model_metrics.json")) as f:
        METRICS = json.load(f)
except Exception:
    METRICS = {}

NORMAL_RANGES = {
    "Total_Bilirubin":            (0.1, 1.2,  "mg/dL"),
    "Direct_Bilirubin":           (0.0, 0.3,  "mg/dL"),
    "Alkaline_Phosphotase":       (44, 147,   "U/L"),
    "Alamine_Aminotransferase":   (7, 56,     "U/L"),
    "Aspartate_Aminotransferase": (10, 40,    "U/L"),
    "Total_Protiens":             (6.0, 8.3,  "g/dL"),
    "Albumin":                    (3.5, 5.5,  "g/dL"),
    "Albumin_and_Globulin_Ratio": (1.1, 2.5,  "ratio"),
}

SENSOR_MAP = [
    ("Total / Direct Bilirubin", "Optical transcutaneous sensor (450-460 nm LED + photodiode) - jaundice-meter principle", "Non-invasive"),
    ("ALT / AST (SGPT / SGOT)", "Electrochemical enzyme biosensor (screen-printed electrode + enzyme layer)", "Micro-fluidic / finger-prick"),
    ("Alkaline Phosphatase", "Amperometric enzyme sensor with pNPP substrate", "Micro-fluidic"),
    ("Total Proteins", "Bio-impedance spectroscopy electrodes", "Wearable patch"),
    ("Albumin", "Potentiometric immuno-sensor / ion-selective electrode", "Micro-fluidic"),
    ("A/G Ratio", "Computed on MCU: Albumin / (Total Protein - Albumin)", "Firmware"),
    ("Age / Gender", "Stored patient profile on device (EEPROM / app)", "User profile"),
]


def build_feature_vector(p):
    age = float(p["age"])
    gender_male = 1 if str(p.get("gender", "male")).strip().lower() in ("male", "m", "1") else 0
    tb = float(p["total_bilirubin"]); db = float(p["direct_bilirubin"])
    alp = float(p["alkaline_phosphotase"])
    alt = float(p["alt"]); ast = float(p["ast"])
    tp = float(p["total_proteins"]); alb = float(p["albumin"])
    ag = float(p["ag_ratio"]) if p.get("ag_ratio") not in (None, "") else (alb / (tp - alb) if tp - alb > 0 else 0.0)
    values = {
        "Age": age, "Gender_Male": gender_male,
        "Total_Bilirubin": tb, "Direct_Bilirubin": db,
        "Alkaline_Phosphotase": alp, "Alamine_Aminotransferase": alt,
        "Aspartate_Aminotransferase": ast, "Total_Protiens": tp,
        "Albumin": alb, "Albumin_and_Globulin_Ratio": ag,
        "Bilirubin_Ratio": db / tb if tb > 0 else 0.0,
        "AST_ALT_Ratio": ast / alt if alt > 0 else 1.0,
        "Protein_Albumin_Diff": tp - alb,
    }
    return np.array([[values[f] for f in FEATURES]])


def predict_dict(payload):
    X = build_feature_vector(payload)
    proba = float(MODEL.predict_proba(X)[0][1])
    label = int(proba >= 0.5)
    risk = "HIGH" if proba >= 0.75 else ("MODERATE" if proba >= 0.45 else "LOW")
    return {
        "risk_probability": round(proba * 100, 1),
        "prediction": label,
        "prediction_text": "Liver disease indicated" if label else "No liver disease indicated",
        "risk_level": risk,
    }


def _api_predict():
    try:
        return jsonify(predict_dict(request.get_json(force=True)))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _api_health():
    return jsonify({"status": "ok", "model": "liver-risk-ensemble-sklearn", "metrics": METRICS})


@app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/<path:path>", methods=["GET", "POST"])
def router(path=""):
    # Single entry point (Vercel-safe): dispatch on the path manually.
    if request.args.get("debug") == "1":
        return jsonify({"request_path": request.path, "path_arg": path,
                        "method": request.method, "url": request.url})
    p = "/" + path.strip("/")
    if p == "/api/predict":
        return _api_predict()
    if p == "/api/health":
        return _api_health()
    return index()


def index():
    result, alerts = None, []
    form = {"age": 52, "gender": "male", "total_bilirubin": 3.9, "direct_bilirubin": 2.0,
            "alkaline_phosphotase": 195, "alt": 27, "ast": 59,
            "total_proteins": 7.3, "albumin": 2.4, "ag_ratio": 0.49}
    if request.method == "POST":
        form = {k: request.form.get(k, "") for k in form}
        try:
            result = predict_dict(form)
            pretty = {"Total_Bilirubin": "Total Bilirubin", "Direct_Bilirubin": "Direct Bilirubin",
                      "Alkaline_Phosphotase": "Alkaline Phosphatase (ALP)", "Alamine_Aminotransferase": "ALT / SGPT",
                      "Aspartate_Aminotransferase": "AST / SGOT", "Total_Protiens": "Total Proteins",
                      "Albumin": "Albumin", "Albumin_and_Globulin_Ratio": "A/G Ratio"}
            checks = {"Total_Bilirubin": form["total_bilirubin"], "Direct_Bilirubin": form["direct_bilirubin"],
                      "Alkaline_Phosphotase": form["alkaline_phosphotase"], "Alamine_Aminotransferase": form["alt"],
                      "Aspartate_Aminotransferase": form["ast"], "Total_Protiens": form["total_proteins"],
                      "Albumin": form["albumin"], "Albumin_and_Globulin_Ratio": form["ag_ratio"]}
            for k, v in checks.items():
                lo, hi, unit = NORMAL_RANGES[k]
                fv = float(v)
                if fv < lo:
                    alerts.append((pretty[k], f"LOW ({fv} {unit}, normal {lo}-{hi} {unit})"))
                elif fv > hi:
                    alerts.append((pretty[k], f"HIGH ({fv} {unit}, normal {lo}-{hi} {unit})"))
        except Exception as e:
            result = {"error": f"Invalid input: {e}"}
    return render_template_string(PAGE, result=result, alerts=alerts, form=form,
                                  sensor_map=SENSOR_MAP, m=METRICS)


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiverGuard — Liver Disease Risk Prediction</title>
<style>
  :root{
    --ink:#0f2e33; --sub:#5b7a80; --line:#e3eef0; --soft:#f4fafb;
    --teal:#0e9aa7; --teal-d:#0b7c88; --sky:#56c1d6; --mint:#e6f6f4;
    --ok:#2fae74; --warn:#e8a13c; --bad:#e05c5c; --card:#ffffff;
    --shadow:0 10px 30px rgba(15,46,51,.08), 0 2px 6px rgba(15,46,51,.04);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,-apple-system,Roboto,Arial,sans-serif;color:var(--ink);
       background:linear-gradient(180deg,#f7fcfd 0%,#eef7f8 55%,#f7fcfd 100%);min-height:100vh}
  a{color:var(--teal-d)}
  .wrap{max-width:1120px;margin:0 auto;padding:0 22px}
  header{padding:38px 0 8px;text-align:center}
  .logo{display:inline-flex;align-items:center;gap:12px}
  .logo .ic{width:52px;height:52px;border-radius:15px;background:linear-gradient(135deg,var(--teal),var(--sky));
       display:flex;align-items:center;justify-content:center;color:#fff;font-size:26px;box-shadow:var(--shadow)}
  .logo h1{font-size:2rem;letter-spacing:.3px}
  .logo h1 span{color:var(--teal-d)}
  .tag{color:var(--sub);margin-top:10px;font-size:.98rem;max-width:720px;margin-left:auto;margin-right:auto;line-height:1.5}
  .badges{margin-top:14px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
  .badge{background:#fff;border:1px solid var(--line);color:var(--teal-d);font-size:.78rem;font-weight:600;
       padding:6px 12px;border-radius:99px;box-shadow:0 2px 6px rgba(15,46,51,.04)}
  .grid{display:grid;grid-template-columns:1.05fr .95fr;gap:22px;margin-top:30px}
  @media(max-width:920px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:26px;box-shadow:var(--shadow)}
  h2{font-size:.95rem;text-transform:uppercase;letter-spacing:1.6px;color:var(--teal-d);margin-bottom:18px;font-weight:700}
  label{display:block;font-size:.8rem;color:var(--sub);margin:11px 0 5px;font-weight:600}
  input,select{width:100%;padding:11px 13px;border-radius:11px;border:1.5px solid var(--line);background:var(--soft);
       color:var(--ink);font-size:.95rem;transition:border .15s,box-shadow .15s}
  input:focus,select:focus{outline:none;border-color:var(--teal);box-shadow:0 0 0 3px rgba(14,154,167,.12);background:#fff}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  button{margin-top:24px;width:100%;padding:15px;border:none;border-radius:13px;font-size:1rem;font-weight:700;
       background:linear-gradient(90deg,var(--teal),var(--sky));color:#fff;cursor:pointer;letter-spacing:.3px;
       box-shadow:0 8px 20px rgba(14,154,167,.25);transition:transform .08s,box-shadow .15s}
  button:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(14,154,167,.32)}
  .res{text-align:center}
  .pct{font-size:3.1rem;font-weight:800;line-height:1;margin-top:6px}
  .gauge{position:relative;height:16px;border-radius:9px;background:linear-gradient(90deg,#2fae74,#e8a13c 50%,#e05c5c);
       margin:24px 0 6px;overflow:visible}
  .gauge .dot{position:absolute;top:50%;width:22px;height:22px;border-radius:50%;background:#fff;
       border:4px solid var(--ink);transform:translate(-50%,-50%);transition:left .5s cubic-bezier(.2,.8,.2,1)}
  .pill{display:inline-block;padding:9px 26px;border-radius:99px;font-weight:800;letter-spacing:1.5px;margin-top:16px;font-size:.95rem}
  .HIGH{background:#fdecec;color:var(--bad);border:1.5px solid var(--bad)}
  .MODERATE{background:#fdf3e2;color:var(--warn);border:1.5px solid var(--warn)}
  .LOW{background:#e9f8f0;color:var(--ok);border:1.5px solid var(--ok)}
  .verdict{margin-top:16px;color:var(--sub);font-size:.94rem;line-height:1.5}
  .alertbox{margin-top:20px;text-align:left}
  .alert{background:#fdf0f0;border-left:4px solid var(--bad);padding:10px 13px;border-radius:9px;margin:8px 0;font-size:.86rem;color:#7a2f2f}
  .full{grid-column:1/-1}
  .stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
  .stat{flex:1;min-width:120px;background:var(--soft);border:1px solid var(--line);border-radius:14px;padding:16px;text-align:center}
  .stat b{font-size:1.5rem;color:var(--teal-d)}
  .stat small{color:var(--sub);display:block;margin-top:4px;font-size:.78rem}
  table{width:100%;border-collapse:collapse;font-size:.87rem}
  th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line)}
  th{color:var(--teal-d);font-size:.75rem;text-transform:uppercase;letter-spacing:1px}
  pre{background:#0f2e33;color:#c9ecef;border-radius:12px;padding:16px;overflow-x:auto;font-size:.8rem;line-height:1.55}
  code.inline{background:var(--mint);border:1px solid var(--line);padding:2px 7px;border-radius:7px;color:var(--teal-d);font-size:.83rem}
  .note{color:var(--sub);font-size:.88rem;line-height:1.55}
  footer{text-align:center;color:#8aa4a9;font-size:.82rem;margin:44px 0 30px;line-height:1.5}
</style>
</head>
<body>
<header class="wrap">
  <div class="logo"><div class="ic">&#129657;</div><h1>Liver<span>Guard</span></h1></div>
  <p class="tag">Enter biosensor readings below and get an instant liver-disease risk assessment from an
     ensemble machine-learning model trained on the Indian Liver Patient Dataset (Kaggle).</p>
  <div class="badges">
    <span class="badge">Ensemble ML</span><span class="badge">Kaggle ILPD · 583 records</span>
    <span class="badge">IoT / ESP32 ready</span><span class="badge">Serverless · Vercel</span>
  </div>
</header>

<div class="wrap grid">
  <div class="card">
    <h2>Biosensor Readings</h2>
    <form method="post">
      <div class="row2">
        <div><label>Age (years)</label><input name="age" value="{{ form.age }}" required></div>
        <div><label>Gender</label>
          <select name="gender">
            <option value="male" {% if form.gender=='male' %}selected{% endif %}>Male</option>
            <option value="female" {% if form.gender=='female' %}selected{% endif %}>Female</option>
          </select></div>
      </div>
      <div class="row2">
        <div><label>Total Bilirubin (mg/dL) · optical sensor</label>
             <input step="any" name="total_bilirubin" value="{{ form.total_bilirubin }}" required></div>
        <div><label>Direct Bilirubin (mg/dL) · optical sensor</label>
             <input step="any" name="direct_bilirubin" value="{{ form.direct_bilirubin }}" required></div>
      </div>
      <div class="row2">
        <div><label>ALT / SGPT (U/L) · enzyme biosensor</label>
             <input step="any" name="alt" value="{{ form.alt }}" required></div>
        <div><label>AST / SGOT (U/L) · enzyme biosensor</label>
             <input step="any" name="ast" value="{{ form.ast }}" required></div>
      </div>
      <div class="row2">
        <div><label>Alkaline Phosphatase (U/L) · enzyme biosensor</label>
             <input step="any" name="alkaline_phosphotase" value="{{ form.alkaline_phosphotase }}" required></div>
        <div><label>Total Proteins (g/dL) · bio-impedance</label>
             <input step="any" name="total_proteins" value="{{ form.total_proteins }}" required></div>
      </div>
      <div class="row2">
        <div><label>Albumin (g/dL) · immuno-sensor</label>
             <input step="any" name="albumin" value="{{ form.albumin }}" required></div>
        <div><label>A/G Ratio · computed on MCU</label>
             <input step="any" name="ag_ratio" value="{{ form.ag_ratio }}" required></div>
      </div>
      <button type="submit">Analyze Liver Risk</button>
    </form>
  </div>

  <div class="card res">
    <h2>Risk Analysis</h2>
    {% if result and result.error %}
      <div class="alert">{{ result.error }}</div>
    {% elif result %}
      <div class="pct" style="color:{% if result.risk_level=='HIGH' %}var(--bad){% elif result.risk_level=='MODERATE' %}var(--warn){% else %}var(--ok){% endif %}">
        {{ result.risk_probability }}%</div>
      <div class="gauge"><div class="dot" style="left:{{ result.risk_probability }}%"></div></div>
      <div class="pill {{ result.risk_level }}">{{ result.risk_level }} RISK</div>
      <div class="verdict"><b>{{ result.prediction_text }}.</b><br>
      {% if result.risk_level == 'HIGH' %}Immediate clinical consultation advised.
      {% elif result.risk_level == 'MODERATE' %}Consult a physician and re-test in 2 weeks.
      {% else %}No action needed; re-screen annually.{% endif %}</div>
      {% if alerts %}
        <div class="alertbox"><b style="font-size:.85rem;color:var(--sub)">OUT-OF-RANGE SENSOR VALUES</b>
          {% for name, msg in alerts %}<div class="alert"><b>{{ name }}:</b> {{ msg }}</div>{% endfor %}
        </div>
      {% endif %}
    {% else %}
      <p class="note" style="margin-top:24px">Fill in the biosensor readings and press
        <b>Analyze Liver Risk</b>.<br><br>The model returns a probability of liver disease from 10
        biochemical parameters — all measurable with the IoT sensor modules listed below.</p>
    {% endif %}
  </div>

  {% if m %}
  <div class="card full">
    <h2>Model Performance (held-out test set)</h2>
    <div class="stats">
      <div class="stat"><b>{{ m.accuracy }}%</b><small>Accuracy</small></div>
      <div class="stat"><b>{{ m.auc }}</b><small>ROC-AUC</small></div>
      <div class="stat"><b>{{ m.recall }}%</b><small>Recall / Sensitivity</small></div>
      <div class="stat"><b>{{ m.f1 }}%</b><small>F1 Score</small></div>
      <div class="stat"><b>{{ m.precision }}%</b><small>Precision</small></div>
      <div class="stat"><b>{{ m.n }}</b><small>Patient records</small></div>
    </div>
    <p class="note">Soft-voting ensemble of Random Forest + Gradient Boosting + SVM + Logistic Regression,
      trained on the <b>Indian Liver Patient Dataset (Kaggle)</b>. High recall ({{ m.recall }}%) means the
      device catches almost every true patient — the key requirement for a screening tool.</p>
  </div>
  {% endif %}

  <div class="card full">
    <h2>IoT Sensor Integration Map</h2>
    <table>
      <tr><th>Parameter</th><th>Sensor Technology</th><th>Form Factor</th></tr>
      {% for param, tech, formf in sensor_map %}
      <tr><td><b>{{ param }}</b></td><td>{{ tech }}</td><td>{{ formf }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <div class="card full">
    <h2>Device Integration API</h2>
    <p class="note" style="margin-bottom:12px">Your microcontroller (ESP32 / Raspberry Pi Pico W) collects
      sensor readings and POSTs them to this server:</p>
    <pre>POST /api/predict
Content-Type: application/json

{
  "age": 52, "gender": "male",
  "total_bilirubin": 3.9, "direct_bilirubin": 2.0,
  "alkaline_phosphotase": 195, "alt": 27, "ast": 59,
  "total_proteins": 7.3, "albumin": 2.4, "ag_ratio": 0.49
}

Response:
{
  "risk_probability": 88.4,
  "prediction": 1,
  "prediction_text": "Liver disease indicated",
  "risk_level": "HIGH"
}</pre>
    <p class="note" style="margin-top:12px">ESP32 (Arduino): use <code class="inline">HTTPClient</code> to POST the
      JSON above, then drive an LED / buzzer / OLED from the <code class="inline">risk_level</code> field.</p>
  </div>
</div>

<footer class="wrap">LiverGuard — academic screening prototype. Not a medical device; clinical diagnosis must be
  confirmed by a physician. Built for an IoT final-year project.</footer>
</body>
</html>
"""

# Vercel's Python runtime auto-detects the Flask `app` object above (WSGI).
# vercel.json rewrites every route to this function.
