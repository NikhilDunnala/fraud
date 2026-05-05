from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained models
lr_model  = joblib.load("lr_model.pkl")
rf_model  = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler    = joblib.load("scaler.pkl")

def predict_transaction(time, v1, v2, amount):
    input_data = pd.DataFrame([[v1, v2, amount, time]],
                              columns=['V1','V2','Amount','Time'])
    input_scaled = scaler.transform(input_data)

    lr  = lr_model.predict_proba(input_scaled)[0][1]
    rf  = rf_model.predict_proba(input_scaled)[0][1]
    xgb = xgb_model.predict_proba(input_scaled)[0][1]

    final = 0.2*lr + 0.2*rf + 0.6*xgb

    if v1 < -1 and v2 < 0:
        final += 0.15
    if v1 < -1.5 and v2 < -1:
        final = max(final, 0.75)
    if amount > 10000:
        final += 0.10
    final = min(final, 1.0)

    if final >= 0.65:
        verdict = "FRAUD"
        color = "red"
    elif final >= 0.3:
        verdict = "UNDER REVIEW"
        color = "orange"
    else:
        verdict = "LEGITIMATE"
        color = "green"

    return {
        "verdict": verdict,
        "color": color,
        "score": round(final * 100, 2),
        "lr": round(lr * 100, 2),
        "rf": round(rf * 100, 2),
        "xgb": round(xgb * 100, 2),
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        time   = float(request.form["time"])
        v1     = float(request.form["v1"])
        v2     = float(request.form["v2"])
        amount = float(request.form["amount"])
        result = predict_transaction(time, v1, v2, amount)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)