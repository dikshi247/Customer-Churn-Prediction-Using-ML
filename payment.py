# payment.py
from flask import Blueprint, request, render_template, redirect, url_for, flash
from pymongo import MongoClient
import os
import datetime

bp = Blueprint("payment", __name__)

# ------------------------------
# MongoDB connection
# ------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "ott_churn")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


# ------------------------------
# Payment Page (GET)
# ------------------------------
@bp.route("/payment", methods=["GET"])
def payment_page():
    plan = request.args.get("plan")
    duration = request.args.get("duration")
    total_price = request.args.get("total_price")
    customer_id = request.args.get("customer_id")

    # Fetch customer name
    customer = db.customers.find_one({"customer_id": customer_id})
    customer_name = customer.get("name") if customer else "Customer"

    return render_template(
        "payment.html",
        plan=plan,
        duration=duration,
        total_price=total_price,
        customer_id=customer_id,
        customer_name=customer_name
    )


# ------------------------------
# Payment Processor (POST)
# ------------------------------
@bp.route("/process_payment", methods=["POST"])
def process_payment():
    customer_id = request.form.get("customer_id")
    plan = request.form.get("plan")
    duration = request.form.get("duration")
    total_price = request.form.get("total_price")
    payment_method = request.form.get("payment_method")

    # Record payment in MongoDB
    db.payments.insert_one({
        "customer_id": customer_id,
        "plan": plan,
        "duration": duration,
        "total_price": total_price,
        "payment_method": payment_method,
        "timestamp": datetime.datetime.utcnow()
    })

    # Update customer's subscription
    db.customers.update_one(
        {"customer_id": customer_id},
        {"$set": {
            "subscription_plan": plan,
            "status": "active",
            "renewal_months": duration,
            "last_payment": datetime.datetime.utcnow()
        }}
    )

    flash("Payment successful! Your subscription is active.", "success")
    return redirect(url_for("customer_dashboard"))
