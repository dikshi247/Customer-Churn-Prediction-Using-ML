# backend/ml/recommend.py
def recommend_from_top_features(top_feature_names):
    """Return a short list of human-friendly recommendations."""
    recs = []
    for f in top_feature_names:
        if "tenure" in f:
            recs.append("Offer a loyalty discount and personalized content — long-term engagement helps.")
        if "MonthlyCharges" in f or "TotalCharges" in f:
            recs.append("Offer flexible pricing or bundle discounts to reduce churn from cost concerns.")
        if "total_watch_minutes" in f or "avg_minutes" in f:
            recs.append("Send personalized recommendations and highlight new content to boost watch time.")
        if "watch_sessions" in f:
            recs.append("Engage with notifications and push marketing to increase session frequency.")
    if not recs:
        recs.append("Offer general retention incentives: coupon, email outreach, and VIP support.")
    # dedupe & keep top 4
    seen = []
    out = []
    for r in recs:
        if r not in seen:
            out.append(r); seen.append(r)
        if len(out) >= 4:
            break
    return out
