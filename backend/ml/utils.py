import os
import pandas as pd

CUSTOMER_DATA_FILE = "backend/ml/data/customers_live.csv"

def ensure_data_folder():
    data_folder = "backend/ml/data"
    os.makedirs(data_folder, exist_ok=True)

    if not os.path.exists(CUSTOMER_DATA_FILE):
        df = pd.DataFrame(columns=[
            "customer_id","gender","senior_citizen","partner","dependents",
            "tenure","phone_service","multiple_lines","internet_service",
            "online_security","online_backup","device_protection",
            "tech_support","streaming_tv","streaming_movies",
            "contract","paperless_billing","payment_method",
            "monthly_charges","total_charges","churn"
        ])
        df.to_csv(CUSTOMER_DATA_FILE, index=False)

def aggregate_customer_features(customer_doc):
    return {
        "customer_id": customer_doc.get("customer_id"),
        "gender": customer_doc.get("gender"),
        "senior_citizen": customer_doc.get("senior_citizen"),
        "partner": customer_doc.get("partner"),
        "dependents": customer_doc.get("dependents"),
        "tenure": customer_doc.get("tenure"),
        "phone_service": customer_doc.get("phone_service"),
        "multiple_lines": customer_doc.get("multiple_lines"),
        "internet_service": customer_doc.get("internet_service"),
        "online_security": customer_doc.get("online_security"),
        "online_backup": customer_doc.get("online_backup"),
        "device_protection": customer_doc.get("device_protection"),
        "tech_support": customer_doc.get("tech_support"),
        "streaming_tv": customer_doc.get("streaming_tv"),
        "streaming_movies": customer_doc.get("streaming_movies"),
        "contract": customer_doc.get("contract"),
        "paperless_billing": customer_doc.get("paperless_billing"),
        "payment_method": customer_doc.get("payment_method"),
        "monthly_charges": customer_doc.get("monthly_charges"),
        "total_charges": customer_doc.get("total_charges"),
        "churn": customer_doc.get("churn")
    }
