import os
import datetime
from pymongo import MongoClient

# Connection string (local MongoDB). Change if you use Atlas.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)

# Database name
db = client["ott_churn"]

# Collections
admins_col = db["admins"]
customers_col = db["customers"]
events_col = db["events"]


# ---------- Admin Helpers ----------

def get_admin_by_email(email: str):
    return admins_col.find_one({"email": email})


def create_admin(email: str, password_hash: str):
    admins_col.insert_one(
        {
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.datetime.utcnow(),
        }
    )


# ---------- Customer Helpers ----------

def get_customer_by_email(email: str):
    return customers_col.find_one({"email": email})


def get_customer_by_id(customer_id: str):
    return customers_col.find_one({"customerID": customer_id})


def create_customer(customer_id: str, name: str, email: str, password_hash: str):
    customers_col.insert_one(
        {
            "customerID": customer_id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "plan": None,
            "status": "inactive",
            "created_at": datetime.datetime.utcnow(),
        }
    )


def update_customer_plan_status(customer_id: str, plan: str | None, status: str):
    update_fields = {"status": status}
    if plan is not None:
        update_fields["plan"] = plan

    customers_col.update_one({"customerID": customer_id}, {"$set": update_fields})


# ---------- Event Logging ----------

def log_customer_event(customer_id: str, event_type: str, extra: dict | None = None):
    event = {
        "customerID": customer_id,
        "event_type": event_type,
        "timestamp": datetime.datetime.utcnow(),
    }
    if extra:
        event.update(extra)
    events_col.insert_one(event)
