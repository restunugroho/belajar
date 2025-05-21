# core.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tqdm import tqdm

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .preprocessing import clean_data


def run_experiment(
    data: pd.DataFrame,
    target: str,
    task: str = "regression",
    split_method: str = "date",
    split_column: str = None,
    test_size: float = 0.2,
    output_dir: str = "output",
    use_gridsearch: bool = True
):
    assert task in ["regression", "classification"], "Invalid task"
    assert split_method in ["random", "date"], "Invalid split method"
    assert target in data.columns, "Target not found in data"

    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()

    # Encode kategorikal
    for col in df.select_dtypes(include="object").columns:
        if col != target:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Split X, y dan bersihkan dengan clean_data()
    X = df.drop(columns=[target])
    y = df[target]
    X, y, datetime_cols = clean_data(X, y)
    dropped = data.shape[0] - X.shape[0]

    # Split train-test
    if split_method == "random":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        assert split_column is not None, "Date column must be provided for date split"
        assert split_column in X.columns, f"{split_column} tidak ada di data"
        X[split_column] = pd.to_datetime(X[split_column])
        X = X.sort_values(by=split_column)
        split_idx = int((1 - test_size) * len(X))
        X_train = X.iloc[:split_idx].drop(columns=split_column)
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:].drop(columns=split_column)
        y_test = y.iloc[split_idx:]

    # Model dan param tuning
    model_dict = {
        "linear": LinearRegression() if task == "regression" else LogisticRegression(max_iter=1000),
        "rf": RandomForestRegressor() if task == "regression" else RandomForestClassifier(),
        "xgb": XGBRegressor() if task == "regression" else XGBClassifier(),
        "cat": CatBoostRegressor(verbose=0) if task == "regression" else CatBoostClassifier(verbose=0),
        "lgbm": LGBMRegressor() if task == "regression" else LGBMClassifier()
    }

    param_grids = {
        "rf": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "xgb": {"n_estimators": [100], "max_depth": [3, 6]},
        "cat": {"depth": [4, 6], "learning_rate": [0.01, 0.1]},
        "lgbm": {"num_leaves": [31, 50], "learning_rate": [0.01, 0.1]}
    }

    results = []

    for name, model in tqdm(model_dict.items(), desc="Training models"):
        best_params = None

        if use_gridsearch and name in param_grids:
            search = GridSearchCV(model, param_grids[name], cv=3, scoring='r2' if task == "regression" else "f1_weighted")
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        if task == "regression":
            train_score = {
                "rmse": mean_squared_error(y_train, y_pred_train, squared=False),
                "r2": r2_score(y_train, y_pred_train)
            }
            test_score = {
                "rmse": mean_squared_error(y_test, y_pred_test, squared=False),
                "r2": r2_score(y_test, y_pred_test)
            }
        else:
            train_score = {
                "accuracy": accuracy_score(y_train, y_pred_train),
                "f1": f1_score(y_train, y_pred_train, average="weighted")
            }
            test_score = {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "f1": f1_score(y_test, y_pred_test, average="weighted")
            }

        results.append({
            "model": name,
            "train_metrics": train_score,
            "test_metrics": test_score,
            "params": best_params
        })

        # SHAP plot
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)
            plt.figure()
            shap.summary_plot(shap_values, X_train, show=False)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_{name}.png")
            plt.close()
        except Exception as e:
            print(f"SHAP failed for {name}: {e}")

    # Tulis laporan
    report_path = os.path.join(output_dir, "report.qmd")
    with open(report_path, "w") as f:
        f.write("# Hasil Eksperimen\n\n")
        f.write(f"Dropped missing rows: {dropped}\n\n")
        f.write(f"Train/Test split method: {split_method}, test size: {test_size}\n\n")
        f.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n\n")

        for res in results:
            f.write(f"## Model: {res['model']}\n")
            if res["params"]:
                f.write("### Best Parameters:\n")
                for k, v in res["params"].items():
                    f.write(f"- {k}: {v}\n")
            f.write("### Train Metrics:\n")
            for k, v in res["train_metrics"].items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write("### Test Metrics:\n")
            for k, v in res["test_metrics"].items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write(f"![SHAP Plot](shap_{res['model']}.png)\n\n")
