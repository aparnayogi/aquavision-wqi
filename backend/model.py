import os
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURES = [
    "ph",
    "dissolved_oxygen",
    "turbidity",
    "conductivity",
    "bod",
    "nitrates",
    "total_coliform",
]


# ── Training ─────────────────────────────────────────────────────────────────

def train(csv_path: str = None) -> dict:
    """
    Train Linear Regression, Random Forest, and XGBoost.
    Save the best model. Return evaluation results for all three.
    """
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "water_quality.csv"
        )

    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y = df["wqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":           XGBRegressor(
                                 n_estimators=100,
                                 learning_rate=0.1,
                                 max_depth=5,
                                 random_state=42,
                                 verbosity=0,
                             ),
    }

    results = {}
    best_model = None
    best_r2 = -999

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = round(r2_score(y_test, y_pred), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

        results[name] = {"r2": r2, "rmse": rmse}
        print(f"{name:22s}  R²={r2:.4f}  RMSE={rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (R²={best_r2:.4f})")
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved → {MODEL_PATH}")

    return {
        "evaluation": results,
        "best_model": best_name,
        "best_r2":    best_r2,
    }


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_wqi(input_dict: dict) -> dict:
    """
    Load saved model and return WQI score, label, and feature importances.
    Falls back to formula if no model is trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        wqi = _fallback_wqi(input_dict)
        return {
            "wqi_score":          round(wqi, 2),
            "wqi_label":          _wqi_label(wqi),
            "feature_importance": _dummy_importance(),
            "model_used":         "formula_fallback",
        }

    model = joblib.load(MODEL_PATH)
    values = [float(input_dict.get(f, 0)) for f in FEATURES]
    X = np.array(values).reshape(1, -1)
    wqi = float(model.predict(X)[0])
    wqi = max(0.0, min(100.0, wqi))

    # ── Feature importance ───────────────────────────────────────────────────
    importance = {}

    if hasattr(model, "feature_importances_"):
        # Random Forest and XGBoost have built-in importances
        raw = model.feature_importances_
        for name, val in zip(FEATURES, raw):
            importance[name] = round(float(val), 4)
    else:
        # Linear Regression: use absolute coefficient values normalised to sum=1
        coefs = np.abs(model.coef_)
        coefs = coefs / coefs.sum()
        for name, val in zip(FEATURES, coefs):
            importance[name] = round(float(val), 4)

    return {
        "wqi_score":          round(wqi, 2),
        "wqi_label":          _wqi_label(wqi),
        "feature_importance": importance,
        "model_used":         type(model).__name__,
    }


def get_shap_values(input_dict: dict) -> dict:
    """
    Compute SHAP values for a single prediction.
    Returns per-feature SHAP contributions.
    """
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet"}

    model = joblib.load(MODEL_PATH)
    values = [float(input_dict.get(f, 0)) for f in FEATURES]
    X = pd.DataFrame([values], columns=FEATURES)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0]
    except Exception:
        # Fallback for Linear Regression
        explainer = shap.LinearExplainer(model, X)
        shap_vals = explainer.shap_values(X)[0]

    return {
        feature: round(float(val), 4)
        for feature, val in zip(FEATURES, shap_vals)
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wqi_label(score: float) -> str:
    if score >= 90:   return "Excellent"
    elif score >= 70: return "Good"
    elif score >= 50: return "Poor"
    elif score >= 25: return "Very Poor"
    return "Unsuitable"


def _fallback_wqi(d: dict) -> float:
    ph_score  = max(0, 100 - abs(d.get("ph", 7) - 7) * 20)
    do_score  = min(100, d.get("dissolved_oxygen", 7) * 12)
    turb_score = max(0, 100 - d.get("turbidity", 1) * 5)
    bod_score = max(0, 100 - d.get("bod", 2) * 10)
    col_score = 100 if d.get("total_coliform", 0) == 0 else 0
    return (ph_score * 0.2 + do_score * 0.25 + turb_score * 0.2 +
            bod_score * 0.2 + col_score * 0.15)


def _dummy_importance() -> dict:
    return {
        "ph": 0.20, "dissolved_oxygen": 0.25, "turbidity": 0.18,
        "conductivity": 0.12, "bod": 0.13, "nitrates": 0.07,
        "total_coliform": 0.05,
    }


# ── Run training directly ─────────────────────────────────────────────────────
if __name__ == "__main__":
    results = train()
    print("\nFull evaluation results:")
    for model_name, metrics in results["evaluation"].items():
        print(f"  {model_name}: R²={metrics['r2']}  RMSE={metrics['rmse']}")