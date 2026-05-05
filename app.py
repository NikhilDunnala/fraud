from flask import Flask, render_template, request, redirect, url_for, flash
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
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "fallback_secret")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# =========================
# 🤖 LOAD MODELS
# =========================
try:
    lr_model  = joblib.load("lr_model.pkl")
    rf_model  = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler    = joblib.load("scaler.pkl")
except Exception as e:
    print("Model loading error:", e)

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

# Create DB
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
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
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
        name = request.form.get("name")
        email = request.form.get("email")
        password = generate_password_hash(request.form.get("password"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already exists")
            return redirect("/signup")

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
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
        file = request.files.get("file")

        if not file or file.filename == "":
            return "❌ No file selected"

        df = pd.read_csv(file)

        # Validate columns
        missing_cols = [col for col in FEATURES if col not in df.columns]
        if missing_cols:
            return f"❌ Missing columns: {missing_cols}"

        X = df[FEATURES].copy()

        # Scale
        X[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

        # Predictions
        prob_lr  = lr_model.predict_proba(X)[:, 1]
        prob_rf  = rf_model.predict_proba(X)[:, 1]
        prob_xgb = xgb_model.predict_proba(X)[:, 1]

        final_prob = 0.2 * prob_lr + 0.2 * prob_rf + 0.6 * prob_xgb

        df['Fraud Probability'] = final_prob
        df['Result'] = df['Fraud Probability'].apply(
            lambda x: "FRAUD" if x >= 0.65 else (
                "UNDER REVIEW" if x >= 0.3 else "LEGIT"
            )
        )

        total = len(df)
        fraud = (df['Result'] == "FRAUD").sum()

        # ✅ CLEAN TABLE (NO \n + ALL ROWS)
        table_html = df.to_html(classes='data', index=False).replace("\\n", "")

        return render_template(
            "result.html",
            tables=[table_html],
            total=total,
            fraud=fraud
        )

    except Exception as e:
        return f"Error: {str(e)}<br><pre>{traceback.format_exc()}</pre>"

# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
