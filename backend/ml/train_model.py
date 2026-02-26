# backend/ml/train_model.py
import pandas as pd, os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

DATA_CSV = "backend/ml/data/current_customers.csv"
FALLBACK_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # placed in project root
MODEL_PATH = "backend/ml/models/churn_model.pkl"
META_PATH = "backend/ml/models/metadata.pkl"

def prepare_df(df):
    # For simplicity: If the df has total_watch_minutes etc, keep them,
    # otherwise attempt to use telco columns.
    # We'll create a consistent set of features:
    features = []
    if "total_watch_minutes" in df.columns:
        # choose a stable list
        features = ["tenure","MonthlyCharges","TotalCharges","total_watch_minutes","watch_sessions","avg_minutes_per_session"]
        X = df[features].fillna(0)
        # no one-hot here
        return X, None
    else:
        # fallback telco simple pipeline
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        X = df[["tenure","MonthlyCharges","TotalCharges"]]
        return X, None

def train_model_if_needed(force=False, data_csv=None):
    # data_csv override
    csv = data_csv or (DATA_CSV if os.path.exists(DATA_CSV) else FALLBACK_CSV)
    if not os.path.exists(csv):
        raise FileNotFoundError(f"No training csv found at {csv}")
    df = pd.read_csv(csv)
    # require label column 'Churn' for training; if not present, create pseudo-target using heuristics (active -> not churn)
    if "Churn" not in df.columns:
        # if customer status cancelled or no activity -> churn=1 else 0
        if "status" in df.columns:
            df["Churn"] = df["status"].map({"cancelled":1,"inactive":1,"active":0}).fillna(0)
        else:
            # no status -> set all 0 (unsafe), but allow training (very weak)
            df["Churn"] = 0
    X, meta = prepare_df(df)
    y = df["Churn"].astype(int)
    if os.path.exists(MODEL_PATH) and not force:
        model = joblib.load(MODEL_PATH)
        return model, meta
    # train/test split
    if len(df) < 5:
        # fallback: we need at least a few rows to train; if too small, still train but be careful
        pass
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # train simple RFC
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # save
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(meta, META_PATH)
    print("Model saved to", MODEL_PATH)
    return clf, meta

def load_model():
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else None
        return clf, meta
    else:
        return train_model_if_needed()

if __name__ == "__main__":
    train_model_if_needed(force=True)
