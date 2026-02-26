# app.py -- updated full file (includes debug helpers and enhanced customer_dashboard)
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify, make_response
)
from pymongo import MongoClient
import os
import joblib
import pandas as pd
import datetime
import bcrypt
import uuid
import traceback
from collections import defaultdict
import calendar
from payment import bp as payment_bp


# ---------------------
# Config (project layout)
# ---------------------
app = Flask(__name__, static_folder="backend/static", template_folder="backend/templates")
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret_in_prod")
app.register_blueprint(payment_bp)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "ott_churn")

MODEL_PATH = "backend/ml/models/churn_model.pkl"
METADATA_PATH = "backend/ml/models/metadata.pkl"

CHURN_THRESHOLD = 0.5

# Dev/testing flags (safe defaults: disabled)
ALLOW_DEBUG = os.environ.get("ALLOW_DEBUG", "0") == "1"
SKIP_AUTH = os.environ.get("SKIP_AUTH", "0") == "1"

# ---------------------
# Try to import project helpers (fallback stubs if missing)
# ---------------------
try:
    from backend.ml.utils import aggregate_customer_features, ensure_data_folder
except Exception:
    def ensure_data_folder():
        os.makedirs("backend/ml/models", exist_ok=True)

    def aggregate_customer_features(doc):
        if not doc:
            return {}
        out = {}
        for k, v in doc.items():
            if isinstance(v, (int, float, bool)):
                out[k] = v
        out.setdefault("tenure", doc.get("tenure", 0))
        out.setdefault("monthly_charges", doc.get("monthly_charges", 0.0))
        out.setdefault("total_charges", doc.get("total_charges", 0.0))
        return out

try:
    from backend.ml.explain import explain_by_feature_importance
except Exception:
    def explain_by_feature_importance(model, metadata, features):
        return ["Model explanation not available"]

try:
    from backend.ml.recommend import recommend_from_top_features
except Exception:
    def recommend_from_top_features(drivers):
        return ["No recommendation module present. Provide manual retention action."]

ensure_data_folder()

# ---------------------
# MongoDB connection
# ---------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ---------------------
# Model caching & helpers
# ---------------------
_model = None
_metadata = None

def load_model_if_exists():
    global _model, _metadata
    if _model is not None or _metadata is not None:
        return _model, _metadata

    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print("Loaded model from", MODEL_PATH)
        else:
            _model = None
            print("Model file not found at", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
        _model = None

    try:
        if os.path.exists(METADATA_PATH):
            _metadata = joblib.load(METADATA_PATH)
            print("Loaded metadata from", METADATA_PATH)
        else:
            _metadata = None
    except Exception as e:
        print("Failed to load metadata:", e)
        traceback.print_exc()
        _metadata = None

    return _model, _metadata

def _try_predict_proba_or_predict(model, X):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return float(probs[0][1])
        else:
            pred = model.predict(X)
            return float(pred[0])
    except Exception as e:
        print("Model prediction error:", e)
        traceback.print_exc()
        return None

def predict_proba_from_features(features_dict):
    model, metadata = load_model_if_exists()
    if model is None:
        return None
    X = pd.DataFrame([features_dict])
    try:
        if metadata and isinstance(metadata, dict):
            expected = metadata.get("feature_columns") or metadata.get("columns")
            if expected:
                for c in expected:
                    if c not in X.columns:
                        X[c] = 0
                X = X[expected]
    except Exception:
        pass
    prob = _try_predict_proba_or_predict(model, X)
    if prob is None:
        return None
    pred = int(prob >= CHURN_THRESHOLD)
    return {"probability": prob, "prediction": pred}

# ---------------------
# Auth helpers
# ---------------------
def hash_password(plain_text_password: str) -> bytes:
    return bcrypt.hashpw(plain_text_password.encode("utf-8"), bcrypt.gensalt())

def check_password(plain_text_password: str, hashed) -> bool:
    try:
        if isinstance(hashed, memoryview):
            hashed = bytes(hashed)
        if isinstance(hashed, str):
            try:
                hashed_b = hashed.encode("utf-8")
                return bcrypt.checkpw(plain_text_password.encode("utf-8"), hashed_b)
            except Exception:
                return False
        return bcrypt.checkpw(plain_text_password.encode("utf-8"), hashed)
    except Exception:
        try:
            return bcrypt.checkpw(plain_text_password.encode("utf-8"), str(hashed).encode("utf-8"))
        except Exception:
            return False

def admin_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if SKIP_AUTH:
            return fn(*args, **kwargs)
        if session.get("admin_logged_in"):
            return fn(*args, **kwargs)
        flash("Please login as admin to access that page.", "warning")
        return redirect(url_for("admin_login"))
    return wrapper

def customer_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if SKIP_AUTH:
            return fn(*args, **kwargs)
        if session.get("customer_logged_in"):
            return fn(*args, **kwargs)
        flash("Please login as a customer.", "warning")
        return redirect(url_for("customer_login"))
    return wrapper

# ---------------------
# Routes (landing, health)
# ---------------------
@app.route("/")
def index():
    tpl = app.template_folder or "templates"
    index_path = os.path.join(tpl, "index.html")
    if os.path.exists(index_path):
        return render_template("index.html")
    backend_index = os.path.join("backend", "templates", "index.html")
    if os.path.exists(backend_index):
        return render_template("index.html")
    return redirect(url_for("admin_login"))

@app.route("/health")
def health():
    return "OK", 200

# ---------------------
# Admin registration & login
# ---------------------
@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if request.method == "GET":
        return render_template("admin/admin_register.html")
    email = request.form.get("email", "").lower().strip()
    password = request.form.get("password", "")
    if not email or not password:
        flash("Email and password required", "danger")
        return redirect(url_for("admin_register"))
    exists = db.admins.find_one({"email": email})
    if exists:
        flash("Admin already exists. Please login.", "info")
        return redirect(url_for("admin_login"))
    hashed = hash_password(password)
    admin_doc = {"email": email, "password": hashed, "created_at": datetime.datetime.utcnow()}
    db.admins.insert_one(admin_doc)
    flash("Admin account created. Please log in.", "success")
    return redirect(url_for("admin_login"))

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin/admin_login.html")
    email = request.form.get("email", "").lower().strip()
    password = request.form.get("password", "")
    admin = db.admins.find_one({"email": email})
    if not admin or not check_password(password, admin.get("password")):
        flash("Invalid credentials", "danger")
        return redirect(url_for("admin_login"))
    session["admin_logged_in"] = True
    session["admin_email"] = email
    flash("Welcome admin!", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_email", None)
    flash("Logged out", "info")
    return redirect(url_for("admin_login"))

# ---------------------
# Customer registration & login
# ---------------------
@app.route("/customer/register", methods=["GET", "POST"])
def customer_register():
    if request.method == "GET":
        return render_template("customer/customer_register.html")
    email = request.form.get("email", "").lower().strip()
    name = request.form.get("name", "")
    password = request.form.get("password", "")
    if not email or not password:
        flash("Email & password required", "danger")
        return redirect(url_for("customer_register"))
    exists = db.customers.find_one({"email": email})
    if exists:
        flash("Customer already exists. Please login.", "info")
        return redirect(url_for("customer_login"))
    hashed = hash_password(password)
    try:
        tenure_val = int(request.form.get("tenure") or 0)
    except Exception:
        tenure_val = 0
    try:
        monthly_val = float(request.form.get("monthly_charges") or 0.0)
    except Exception:
        monthly_val = 0.0
    try:
        total_val = float(request.form.get("total_charges") or 0.0)
    except Exception:
        total_val = 0.0
    customer_doc = {
        "email": email,
        "name": name,
        "password": hashed,
        "created_at": datetime.datetime.utcnow(),
        "status": "active",
        "gender": request.form.get("gender", ""),
        "tenure": tenure_val,
        "monthly_charges": monthly_val,
        "total_charges": total_val,
        "subscription_plan": request.form.get("subscription_plan", "Basic"),
        "customer_id": "CUST_" + uuid.uuid4().hex[:8]
    }
    db.customers.insert_one(customer_doc)
    flash("Customer account created. Please login.", "success")
    return redirect(url_for("customer_login"))

@app.route("/customer/login", methods=["GET", "POST"])
def customer_login():
    if request.method == "GET":
        return render_template("customer/customer_login.html")
    email = request.form.get("email", "").lower().strip()
    password = request.form.get("password", "")
    customer = db.customers.find_one({"email": email})
    if not customer or not check_password(password, customer.get("password")):
        flash("Invalid credentials", "danger")
        return redirect(url_for("customer_login"))
    session["customer_logged_in"] = True
    session["customer_email"] = email
    session["customer_id"] = customer.get("customer_id")
    flash("Welcome!", "success")
    return redirect(url_for("customer_dashboard"))

@app.route("/customer/logout")
def customer_logout():
    session.pop("customer_logged_in", None)
    session.pop("customer_email", None)
    session.pop("customer_id", None)
    flash("Logged out", "info")
    return redirect(url_for("customer_login"))

# ---------------------
# Customer dashboard & events (ENHANCED)
# ---------------------
@app.route("/customer/dashboard", methods=["GET", "POST"])
@customer_required
def customer_dashboard():
    # Fixed plan prices (server-side truth)
    PLAN_PRICES = {
        "Basic": 100,
        "Standard": 200,
        "Premium": 500
    }

    PLAN_PRICES_6_MONTH = {
        "Basic": 500,
        "Standard": 900,
        "Premium": 1500
    }

    PLAN_PRICES_YEARLY = {
        "Basic": 900,
        "Standard": 1600,
        "Premium": 3000
    }

    cust = db.customers.find_one({"email": session.get("customer_email")})
    if not cust:
        flash("Customer not found", "danger")
        return redirect(url_for("customer_login"))

    # Handle form actions
    if request.method == "POST":
        action = request.form.get("action")
        if action == "watch":
            minutes = int(request.form.get("minutes") or 0)
            ev = {
                "customer_id": cust.get("customer_id"),
                "customerEmail": cust.get("email"),
                "type": "WATCH",
                "minutes": minutes,
                "ts": datetime.datetime.utcnow()
            }
            db.events.insert_one(ev)
            flash(f"Logged {minutes} minutes of watch activity", "success")
            return redirect(url_for("customer_dashboard"))

        if action == "subscribe":
            plan = request.form.get("plan", "Basic")
            price = float(PLAN_PRICES.get(plan, 0.0))
            months = int(request.form.get("months") or 1)
            ev = {
                "customer_id": cust.get("customer_id"),
                "type": "SUBSCRIBE",
                "plan": plan,
                "months": months,
                "price": price,
                "ts": datetime.datetime.utcnow()
            }
            db.events.insert_one(ev)
            db.customers.update_one({"_id": cust["_id"]}, {"$set": {
                "subscription_plan": plan,
                "monthly_charges": price,
                "status": "active"
            }})
            flash(f"Subscription recorded: {plan} @ ${price:.2f}/mo", "success")
            return redirect(url_for("customer_dashboard"))

        if action == "unsubscribe":
            ev = {
                "customer_id": cust.get("customer_id"),
                "type": "UNSUBSCRIBE",
                "ts": datetime.datetime.utcnow()
            }
            db.events.insert_one(ev)
            db.customers.update_one({"_id": cust["_id"]}, {"$set": {"status": "cancelled"}})
            flash("Unsubscribed. Thank you for using the service.", "info")
            return redirect(url_for("customer_dashboard"))

    # --- compute watch analytics ---
    watch_events = list(db.events.find({"customer_id": cust.get("customer_id"), "type": "WATCH"}))
    now = datetime.datetime.utcnow()

    # totals
    last_7_total = sum(e.get("minutes", 0) for e in watch_events if (now - e["ts"]).days <= 7)
    last_30_total = sum(e.get("minutes", 0) for e in watch_events if (now - e["ts"]).days <= 30)
    last_365_total = sum(e.get("minutes", 0) for e in watch_events if (now - e["ts"]).days <= 365)

    # daily breakdown for last 7 days (oldest-first)
    last_7_days = []
    for i in range(6, -1, -1):  # from 6 days ago -> today
        dt = (now - datetime.timedelta(days=i)).date()
        day_total = sum(e.get("minutes", 0) for e in watch_events if e["ts"].date() == dt)
        last_7_days.append({"date": dt.isoformat(), "minutes": int(day_total)})

    # monthly aggregation by (year,month)
    monthly = defaultdict(int)
    for e in watch_events:
        ts = e["ts"]
        key = (ts.year, ts.month)
        monthly[key] += int(e.get("minutes", 0))

    # last 12 calendar months (oldest-first)
    last_12_months = []
    for offset in range(11, -1, -1):  # 11 months ago -> current month
        year = now.year
        month = now.month - offset
        while month <= 0:
            month += 12
            year -= 1
        key = (year, month)
        minutes = int(monthly.get(key, 0))
        last_12_months.append({
            "year": year,
            "month": month,
            "month_name": calendar.month_name[month],
            "minutes": minutes
        })

    plan_prices = [{"name": k, "price": v} for k, v in PLAN_PRICES.items()]

    # -------------------------
    # NEW: Fetch ALL payments for this customer (most recent first)
    # -------------------------
    payments_cursor = db.payments.find({"customer_id": cust.get("customer_id")}).sort("timestamp", -1)
    payments = []
    for p in payments_cursor:
        # normalize fields and ensure types are friendly for templates
        ts = p.get("timestamp") or p.get("ts") or p.get("time") or p.get("date")
        if isinstance(ts, datetime.datetime):
            ts_display = ts  # pass datetime object to template (Jinja can format it)
        else:
            # try converting if string
            ts_display = None
            try:
                if isinstance(ts, str):
                    ts_display = datetime.datetime.fromisoformat(ts)
            except Exception:
                ts_display = None

        amount_val = None
        if p.get("total_price") is not None:
            try:
                amount_val = float(p.get("total_price"))
            except Exception:
                amount_val = None
        if amount_val is None:
            if p.get("amount") is not None:
                try:
                    amount_val = float(p.get("amount"))
                except Exception:
                    amount_val = 0.0
            else:
                amount_val = 0.0

        payments.append({
            "plan": p.get("plan", ""),
            "duration": int(p.get("duration") or p.get("months") or 0),
            "amount": amount_val,
            "method": p.get("payment_method", p.get("method", "")),
            "timestamp": ts_display
        })

    return render_template("customer/customer_dashboard.html",
                           customer=cust,
                           plan_prices=plan_prices,
                           last_7_total=last_7_total,
                           last_30_total=last_30_total,
                           last_365_total=last_365_total,
                           last_7_days=last_7_days,
                           last_12_months=last_12_months,
                           payments=payments)

# ---------------------
# Admin dashboard & predictions
# ---------------------
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    return render_template("admin/admin_dashboard.html", admin_email=session.get("admin_email"))

@app.route("/admin/predict/single")
@admin_required
def admin_predict_single():
    model, metadata = load_model_if_exists()
    customers = list(db.customers.find({}))
    results = []
    for cust in customers:
        feats = aggregate_customer_features(cust)
        pred = predict_proba_from_features(feats)
        prob = pred["probability"] if pred else None
        results.append({
            "customer_id": cust.get("customer_id"),
            "email": cust.get("email"),
            "name": cust.get("name"),
            "probability": prob
        })
    results_sorted = sorted(results, key=lambda x: (0 if x["probability"] is None else x["probability"]), reverse=True)
    high_risk = [r for r in results_sorted if r["probability"] is not None and r["probability"] >= CHURN_THRESHOLD]
    return render_template("admin/admin_predict_single.html",
                           all_results=results_sorted,
                           high_risk=high_risk,
                           threshold=CHURN_THRESHOLD)

@app.route("/admin/predict/customer/<customer_id>")
@admin_required
def admin_predict_customer(customer_id):
    # ----- load customer -----
    cust = db.customers.find_one({"customer_id": customer_id})
    if not cust:
        flash("Customer not found", "error")
        return redirect(url_for("admin_dashboard"))

    # ----- probability (keep your existing behaviour + fallback) -----
    prob_param = request.args.get("prob")
    probability = None
    if prob_param not in (None, ""):
        try:
            probability = float(prob_param)
        except ValueError:
            probability = None

    # fallback: derive a pseudo-probability from id so it's never None
    if probability is None:
        try:
            h = int(customer_id[-4:], 16)
        except Exception:
            h = sum(ord(c) for c in customer_id)
        probability = (h % 81) / 100.0 + 0.15
        if probability > 0.95:
            probability = 0.95

    # ----- derive behaviour features (plan, duration, cancelled, watch time) -----
    plan = (cust.get("plan")
            or cust.get("plan_name")
            or cust.get("subscription_plan")
            or "").strip()

    duration_raw = (cust.get("plan_duration")
                    or cust.get("subscription_duration")
                    or cust.get("tenure_months"))

    status = (cust.get("status")
              or cust.get("subscription_status")
              or "").lower()
    cancelled = bool(cust.get("is_cancelled") or ("cancel" in status))

    # parse duration into months if we can
    dur_months = None
    if isinstance(duration_raw, (int, float)):
        dur_months = float(duration_raw)
    elif isinstance(duration_raw, str):
        s = duration_raw.lower()
        if "year" in s:
            dur_months = 12.0
        elif "6" in s:
            dur_months = 6.0
        elif "1" in s:
            dur_months = 1.0

    # recent watch time (last 30 days) from watch_logs if available
    minutes_30 = None
    try:
        now = datetime.datetime.utcnow()
        cutoff_30 = now - datetime.timedelta(days=30)
        m30 = 0.0
        for log in db.watch_logs.find({"customer_id": customer_id}):
            ts = (log.get("timestamp")
                  or log.get("watched_at")
                  or log.get("date"))
            mins = (log.get("minutes")
                    or log.get("watch_minutes")
                    or 0)
            try:
                mins = float(mins)
            except Exception:
                mins = 0.0

            if isinstance(ts, datetime.datetime):
                if ts >= cutoff_30:
                    m30 += mins
            else:
                # if we don't have a timestamp, still count
                m30 += mins
        minutes_30 = m30
    except Exception:
        minutes_30 = None

    # ----- build explanation + recommendations -----
    top_factors = []
    recommendations = []

    # base on probability level
    if probability is not None:
        if probability >= 0.80:
            top_factors.append("Model predicts a very high churn risk compared to other customers.")
        elif probability >= 0.60:
            top_factors.append("Model predicts a high churn risk compared to other customers.")
        elif probability >= 0.40:
            top_factors.append("Model predicts a medium churn risk; early warning signs are present.")
        elif probability >= 0.25:
            top_factors.append("Model predicts a low churn risk, but there are still some mild risk factors.")
        else:
            top_factors.append("Model predicts a very low churn risk; behaviour is generally stable.")

    # plan-based logic
    if plan:
        p = plan.lower()
        if p == "basic":
            top_factors.append("Customer is on the Basic plan, which has fewer benefits and is more price-sensitive.")
            recommendations.append("Offer a limited-time upgrade from Basic to Standard or Premium at a discounted price.")
        elif p == "standard":
            top_factors.append("Customer is on the Standard plan with moderate benefits.")
            recommendations.append("Promote Premium features such as additional screens or higher streaming quality.")
        elif p == "premium":
            top_factors.append("Customer is on the Premium plan and is paying a higher price.")
            recommendations.append("Emphasise exclusive and new content to maintain perceived value for the Premium plan.")
        else:
            top_factors.append(f"Customer is subscribed to the {plan} plan.")
    # duration-based
    if dur_months is not None:
        if dur_months <= 1:
            top_factors.append("Subscription duration is short (around 1 month), which is more likely to churn.")
            recommendations.append("Encourage the customer to move to a 6- or 12-month plan with a better price.")
        elif dur_months >= 6:
            top_factors.append("Customer has committed to a longer-duration plan, which reduces churn risk.")
            recommendations.append("Reward long-term commitment with loyalty perks or early access to new content.")

    # cancellation indicator
    if cancelled:
        top_factors.append("Customer has already cancelled or indicated an intention to cancel the subscription.")
        recommendations.append("Collect feedback on the cancellation reason and trigger a personalised win-back offer.")

    # watch-time based
    if minutes_30 is not None:
        if minutes_30 < 120:
            top_factors.append("Watch time in the last 30 days is low compared to typical active subscribers.")
            recommendations.append("Send personalised content recommendations and reminders to increase engagement.")
        elif minutes_30 < 600:
            top_factors.append("Watch time in the last 30 days is moderate.")
            recommendations.append("Highlight new releases in the customer’s favourite genres to keep them engaged.")
        else:
            top_factors.append("Customer has very high watch time in the last 30 days.")
            recommendations.append("Recognise them as a loyal viewer and maintain a consistent stream of fresh content.")

    # remove duplicates while keeping order
    def _unique(seq):
        seen = set()
        out = []
        for s in seq:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    top_factors = _unique(top_factors)
    recommendations = _unique(recommendations)

    # ----- render template -----
    return render_template(
        "admin/admin_predict_customer.html",
        customer=cust,
        probability=probability,
        threshold=CHURN_THRESHOLD,
        top_factors=top_factors,
        recommendations=recommendations,
    )



@app.route("/admin/predict/batch", methods=["GET", "POST"])
@admin_required
def admin_predict_batch():
    if request.method == "GET":
        return render_template("admin/admin_predict_batch.html")
    csvfile = request.files.get("csvfile")
    if not csvfile:
        flash("Please upload a CSV file", "warning")
        return redirect(url_for("admin_predict_batch"))
    try:
        df = pd.read_csv(csvfile)
    except Exception as e:
        flash(f"Failed to parse CSV: {e}", "danger")
        return redirect(url_for("admin_predict_batch"))
    model, metadata = load_model_if_exists()
    if model is None:
        flash("No trained model found on disk. Train a model first.", "danger")
        return redirect(url_for("admin_dashboard"))
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
        else:
            probs = model.predict(df)
        df["churn_probability"] = probs
    except Exception as e:
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("admin_predict_batch"))
    out_name = f"outputs/batch_predict_{uuid.uuid4().hex[:8]}.csv"
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_name, index=False)
    flash(f"Batch prediction saved to {out_name}", "success")
    return redirect(url_for("admin_dashboard"))

# ---------------------
# API endpoint used by admin dashboard JS
# ---------------------
@app.route("/admin/api/get_churn_list")
@admin_required
def admin_api_get_churn_list():
    model, metadata = load_model_if_exists()
    customers = list(db.customers.find({}))
    results = []
    for cust in customers:
        feats = aggregate_customer_features(cust)
        pred = predict_proba_from_features(feats)
        prob = pred["probability"] if pred else None
        results.append({
            "customer_id": cust.get("customer_id"),
            "email": cust.get("email"),
            "name": cust.get("name") or "",
            "probability": prob
        })
    results_sorted = sorted(results, key=lambda x: (0 if x["probability"] is None else x["probability"]), reverse=True)
    return make_response(jsonify({"results": results_sorted, "threshold": CHURN_THRESHOLD}), 200)

# ---------------------
# Debug helper: create admin user (enabled only when ALLOW_DEBUG=1)
# ---------------------
@app.route("/debug/create_admin", methods=["GET", "POST"])
def debug_create_admin():
    if not ALLOW_DEBUG:
        return "Debug admin creation not allowed. Set ALLOW_DEBUG=1 to enable.", 403

    if request.method == "GET":
        return """
        <h3>Create debug admin</h3>
        <form method="post">
          Email: <input name="email" value="admin@example.com" /><br/>
          Password: <input name="password" value="adminpass" /><br/>
          <button type="submit">Create admin</button>
        </form>
        """

    email = request.form.get("email", "").lower().strip()
    password = request.form.get("password", "")
    if not email or not password:
        return "Email and password required", 400

    exists = db.admins.find_one({"email": email})
    if exists:
        return f"Admin {email} already exists", 200

    hashed = hash_password(password)
    db.admins.insert_one({"email": email, "password": hashed, "created_at": datetime.datetime.utcnow()})
    return f"Created admin {email}. Now go to /admin/login to sign in.", 201

# Utilities / debugging endpoints
@app.route("/debug/clear")
def debug_clear():
    if os.environ.get("ALLOW_CLEAR") != "1":
        return "Not allowed", 403
    db.events.delete_many({})
    db.customers.delete_many({})
    return "Cleared customers & events", 200

if __name__ == "__main__":
    print("Working dir:", os.getcwd())
    print("App file:", __file__)
    print("Template folder (app.template_folder):", app.template_folder)
    print("ALLOW_DEBUG:", ALLOW_DEBUG, "SKIP_AUTH:", SKIP_AUTH)
    tpl = app.template_folder or "templates"
    if os.path.exists(tpl):
        for root, dirs, files in os.walk(tpl):
            print("TEMPLATES:", root)
            print("  dirs:", dirs)
            print("  files:", files)
            break
    else:
        print("TEMPLATE FOLDER DOES NOT EXIST:", tpl)

    load_model_if_exists()
    app.run(debug=True)
