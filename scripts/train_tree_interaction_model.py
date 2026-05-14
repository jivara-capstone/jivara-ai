"""
Train a tree-based drug-food interaction recommender.

This is the safer baseline for the current dataset because
drug_food_interactions.csv is rule/curation-generated and very small
(61 foods x 14 drug categories). The model learns from drug_category plus
multi-hot active ingredient features, then predicts severity on a 0-5 scale.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

INTERACTIONS_PATH = DATA_DIR / "drug_food_interactions.csv"
FOOD_KB_PATH = DATA_DIR / "food_to_ingredient_kb.json"
KERAS_ARTIFACTS_PATH = MODEL_DIR / "recommender_artifacts.pkl"
OUTPUT_PATH = MODEL_DIR / "drug_interaction_tree_model.pkl"


ACTIVE_KEYWORDS = [
    "bawang putih",
    "jahe",
    "kunyit",
    "lengkuas",
    "kangkung",
    "bayam",
    "kemangi",
    "daun bawang",
    "gula pasir",
    "gula merah",
    "gula aren",
    "kental manis",
    "madu",
    "pisang",
    "kentang",
    "santan",
    "jeruk",
    "susu",
    "kecap",
    "petis",
    "terasi",
    "tempe",
    "tahu",
    "cuka",
    "kopi",
]


def severity_to_risk(severity: float) -> str:
    if severity == 0:
        return "aman"
    if severity <= 2:
        return "ringan"
    if severity <= 3:
        return "sedang"
    return "tinggi"


def pred_to_risk(severity: float) -> str:
    if severity < 1.0:
        return "aman"
    if severity < 2.5:
        return "ringan"
    if severity < 3.5:
        return "sedang"
    return "tinggi"


def get_ingredient_features(food_ingredients: list[str], keywords: list[str]) -> list[float]:
    return [
        1.0 if any(keyword.lower() in ingredient.lower() for ingredient in food_ingredients) else 0.0
        for keyword in keywords
    ]


def load_food_to_ingredients() -> dict[str, list[str]]:
    with FOOD_KB_PATH.open("r", encoding="utf-8") as file:
        food_kb = json.load(file)
    return food_kb["food_to_ingredients"]


def load_drug_categories() -> dict:
    if not KERAS_ARTIFACTS_PATH.exists():
        return {}
    with KERAS_ARTIFACTS_PATH.open("rb") as file:
        artifacts = pickle.load(file)
    return artifacts.get("drug_categories", {})


def build_feature_frame(
    df: pd.DataFrame,
    food_to_ingredients: dict[str, list[str]],
    active_keywords: list[str] = ACTIVE_KEYWORDS,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        food_name = row["food_class"]
        ingredients = food_to_ingredients.get(food_name, [])
        ingredient_features = get_ingredient_features(ingredients, active_keywords)

        feature_row = {
            "food_class": food_name,
            "drug_category": row["drug_category"],
        }
        feature_row.update(
            {f"kw_{keyword}": value for keyword, value in zip(active_keywords, ingredient_features)}
        )
        rows.append(feature_row)
    return pd.DataFrame(rows)


def make_pipeline(keyword_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("drug_category", OneHotEncoder(handle_unknown="ignore"), ["drug_category"]),
            ("ingredient_features", "passthrough", keyword_columns),
        ],
        remainder="drop",
    )

    model = ExtraTreesRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate_cv(X: pd.DataFrame, y: np.ndarray, risk_labels: np.ndarray, keyword_columns: list[str]) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_true = []
    all_pred = []
    all_true_risk = []
    all_pred_risk = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, risk_labels), start=1):
        pipeline = make_pipeline(keyword_columns)
        pipeline.fit(X.iloc[train_idx], y[train_idx])

        pred = np.clip(pipeline.predict(X.iloc[val_idx]), 0.0, 5.0)
        pred_risk = np.array([pred_to_risk(value) for value in pred])
        true_risk = risk_labels[val_idx]

        mae = mean_absolute_error(y[val_idx], pred)
        rmse = np.sqrt(mean_squared_error(y[val_idx], pred))
        risk_acc = accuracy_score(true_risk, pred_risk)

        fold_metrics.append(
            {
                "fold": fold,
                "mae": float(mae),
                "rmse": float(rmse),
                "risk_accuracy": float(risk_acc),
            }
        )
        all_true.extend(y[val_idx].tolist())
        all_pred.extend(pred.tolist())
        all_true_risk.extend(true_risk.tolist())
        all_pred_risk.extend(pred_risk.tolist())

        print(f"Fold {fold}: MAE={mae:.4f} | RMSE={rmse:.4f} | Risk Acc={risk_acc:.4f}")

    summary = {
        "folds": fold_metrics,
        "mae_mean": float(np.mean([item["mae"] for item in fold_metrics])),
        "mae_std": float(np.std([item["mae"] for item in fold_metrics])),
        "rmse_mean": float(np.mean([item["rmse"] for item in fold_metrics])),
        "rmse_std": float(np.std([item["rmse"] for item in fold_metrics])),
        "risk_accuracy_mean": float(np.mean([item["risk_accuracy"] for item in fold_metrics])),
        "risk_accuracy_std": float(np.std([item["risk_accuracy"] for item in fold_metrics])),
        "classification_report": classification_report(
            all_true_risk,
            all_pred_risk,
            labels=["aman", "ringan", "sedang", "tinggi"],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix_labels": ["aman", "ringan", "sedang", "tinggi"],
        "confusion_matrix": confusion_matrix(
            all_true_risk,
            all_pred_risk,
            labels=["aman", "ringan", "sedang", "tinggi"],
        ).tolist(),
    }
    return summary


def train_and_save() -> dict:
    df = pd.read_csv(INTERACTIONS_PATH)
    food_to_ingredients = load_food_to_ingredients()
    drug_categories = load_drug_categories()

    X = build_feature_frame(df, food_to_ingredients)
    y = df["severity"].astype(float).to_numpy()
    risk_labels = np.array([severity_to_risk(value) for value in y])
    keyword_columns = [f"kw_{keyword}" for keyword in ACTIVE_KEYWORDS]

    print("Training ExtraTreesRegressor drug-food severity model")
    print(f"Rows          : {len(df)}")
    print(f"Foods         : {df['food_class'].nunique()}")
    print(f"Drug category : {df['drug_category'].nunique()}")
    print(f"Features      : drug_category + {len(keyword_columns)} ingredient keywords")
    print()

    metrics = evaluate_cv(X, y, risk_labels, keyword_columns)

    final_pipeline = make_pipeline(keyword_columns)
    final_pipeline.fit(X, y)

    final_pred = np.clip(final_pipeline.predict(X), 0.0, 5.0)
    final_pred_risk = np.array([pred_to_risk(value) for value in final_pred])
    final_metrics = {
        "mae": float(mean_absolute_error(y, final_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, final_pred))),
        "risk_accuracy": float(accuracy_score(risk_labels, final_pred_risk)),
    }

    bundle = {
        "model_type": "ExtraTreesRegressor",
        "pipeline": final_pipeline,
        "active_keywords": ACTIVE_KEYWORDS,
        "keyword_columns": keyword_columns,
        "food_classes": sorted(df["food_class"].unique().tolist()),
        "drug_categories": drug_categories,
        "metrics_cv": metrics,
        "metrics_train_full": final_metrics,
        "severity_scale": "0-5",
        "risk_thresholds": {
            "aman": "< 1.0",
            "ringan": ">= 1.0 and < 2.5",
            "sedang": ">= 2.5 and < 3.5",
            "tinggi": ">= 3.5",
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(bundle, file)

    print()
    print("Cross-validation summary")
    print(f"MAE          : {metrics['mae_mean']:.4f} (+/- {metrics['mae_std']:.4f})")
    print(f"RMSE         : {metrics['rmse_mean']:.4f} (+/- {metrics['rmse_std']:.4f})")
    print(
        "Risk Accuracy: "
        f"{metrics['risk_accuracy_mean'] * 100:.2f}% "
        f"(+/- {metrics['risk_accuracy_std'] * 100:.2f}%)"
    )
    print()
    print("Full-training fit")
    print(f"MAE          : {final_metrics['mae']:.4f}")
    print(f"RMSE         : {final_metrics['rmse']:.4f}")
    print(f"Risk Accuracy: {final_metrics['risk_accuracy'] * 100:.2f}%")
    print()
    print(f"Saved model  : {OUTPUT_PATH}")

    return bundle


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    train_and_save()
