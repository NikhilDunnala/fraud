from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# 🔥 LOAD MODELS
# =========================
lr_model  = joblib.load("lr_model.pkl")
rf_model  = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler    = joblib.load("scaler.pkl")

# Expected feature order
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


# =========================
# 🏠 HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# 📤 UPLOAD & PREDICT
# =========================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    # ❌ No file selected
    if not file or file.filename == "":
        return "❌ Please select a CSV file"

    try:
        df = pd.read_csv(file)
    except:
        return "❌ Invalid CSV file"

    # =========================
    # ✅ VALIDATE COLUMNS
    # =========================
    missing_cols = [col for col in FEATURES if col not in df.columns]

    if missing_cols:
        return f"❌ Missing columns: {missing_cols}"

    # =========================
    # 🧠 PREPARE DATA
    # =========================
    X = df[FEATURES].copy()

    # Scale only Time & Amount
    X[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

    # =========================
    # 🤖 MODEL PREDICTIONS
    # =========================
    prob_lr  = lr_model.predict_proba(X)[:, 1]
    prob_rf  = rf_model.predict_proba(X)[:, 1]
    prob_xgb = xgb_model.predict_proba(X)[:, 1]

    # =========================
    # 🔥 ENSEMBLE
    # =========================
    final_prob = 0.2 * prob_lr + 0.2 * prob_rf + 0.6 * prob_xgb

    # =========================
    # 📊 RESULTS
    # =========================
    df['Fraud Probability'] = final_prob

    df['Result'] = df['Fraud Probability'].apply(
        lambda x: "FRAUD" if x >= 0.65 else (
            "UNDER REVIEW" if x >= 0.3 else "LEGIT"
        )
    )

    # Summary
    total = len(df)
    fraud = (df['Result'] == "FRAUD").sum()

    # =========================
    # 🎯 RETURN OUTPUT
    # =========================
    return render_template(
        "result.html",
        tables=[df.head(20).to_html(classes='data', index=False)],
        total=total,
        fraud=fraud
    )


# =========================
# ▶️ RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
