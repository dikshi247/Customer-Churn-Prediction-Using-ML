# backend/ml/explain.py
def explain_by_feature_importance(model, feature_names, row_series):
    """
    Returns a list of tuples (feature_name, contribution_score) sorted by importance for this row.
    Uses model.feature_importances_ if available and multiplies by feature value.
    """
    if not hasattr(model, "feature_importances_"):
        # fall back to zero importances
        return [(f, 0) for f in feature_names]
    importances = model.feature_importances_
    # pair
    pairs = []
    for fname, imp in zip(feature_names, importances):
        val = float(row_series.get(fname, 0))
        score = imp * (abs(val) + 1)
        pairs.append((fname, round(score,4)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs
