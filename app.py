from flask import Flask, render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)

# =========================
# 🔐 CONFIG (RENDER SAFE)
# =========================
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "secret123")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# =========================
# 🤖 LOAD MODELS (SAFE)
# =========================
lr_model = None
rf_model = None
xgb_model = None
scaler = None

try:
    lr_model = joblib.load("lr_model.pkl")
    print("✅ LR loaded")
except Exception as e:
    print("❌ LR load failed:", e)

try:
    rf_model = joblib.load("rf_model.pkl")
    print("✅ RF loaded")
except Exception as e:
    print("❌ RF load failed:", e)

try:
    xgb_model = joblib.load("xgb_model.pkl")
    print("✅ XGB loaded")
except Exception as e:
    print("❌ XGB load failed:", e)

try:
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler loaded")
except Exception as e:
    print("❌ Scaler load failed:", e)

FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# =========================
# 👤 USER MODEL
# =========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# =========================
# 🏠 HOME
# =========================
@app.route("/")
@login_required
def home():
    return render_template("index.html", user=current_user)

# =========================
# 🔐 LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form.get("email")).first()

        if user and check_password_hash(user.password, request.form.get("password")):
            login_user(user)
            return redirect("/")
        else:
            flash("Invalid credentials")

    return render_template("login.html")

# =========================
# 📝 SIGNUP
# =========================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")

        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect("/signup")

        user = User(
            name=request.form.get("name"),
            email=email,
            password=generate_password_hash(request.form.get("password"))
        )

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("signup.html")

# =========================
# 🚪 LOGOUT
# =========================
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# =========================
# 📤 UPLOAD & PREDICT
# =========================
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    try:
        if scaler is None or (lr_model is None and rf_model is None and xgb_model is None):
            return "❌ Models not loaded properly. Check logs."

        file = request.files.get("file")

        if not file or file.filename == "":
            return "❌ No file selected"

        df = pd.read_csv(file)

        # Validate columns
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return f"❌ Missing columns: {missing}"

        X = df[FEATURES].copy()

        # Scale
        X[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

        # Predictions (safe fallback if any model missing)
        probs = []

        if lr_model:
            probs.append(0.2 * lr_model.predict_proba(X)[:, 1])

        if rf_model:
            probs.append(0.2 * rf_model.predict_proba(X)[:, 1])

        if xgb_model:
            probs.append(0.6 * xgb_model.predict_proba(X)[:, 1])

        final_prob = sum(probs)

        df['Fraud Probability'] = final_prob
        df['Result'] = df['Fraud Probability'].apply(
            lambda x: "FRAUD" if x >= 0.65 else (
                "UNDER REVIEW" if x >= 0.3 else "LEGIT"
            )
        )

        # Counts
        total = len(df)
        fraud = (df['Result'] == "FRAUD").sum()
        review = (df['Result'] == "UNDER REVIEW").sum()
        legit = (df['Result'] == "LEGIT").sum()

        # Clean table
        table_html = df.to_html(classes='table table-striped table-bordered', index=False).replace("\\n", "")

        return render_template(
            "result.html",
            tables=[table_html],
            total=total,
            fraud=fraud,
            review=review,
            legit=legit
        )

    except Exception as e:
        return f"Error: {str(e)}<br><pre>{traceback.format_exc()}</pre>"

# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
