import pandas as pd
import numpy as np
import os
import shutil
import datetime
import uuid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import subprocess

def run_experiment(
    data: pd.DataFrame,
    target_column: str,
    task: str = "regression",
    split_method: str = "date",  # or "random"
    date_column: str = None,
    test_size: float = 0.2,
    output_dir: str = "output"
):
    assert task in ["regression", "classification"], "Task must be 'regression' or 'classification'"
    assert split_method in ["random", "date"], "split_method must be 'random' or 'date'"
    assert target_column in data.columns, f"Target column '{target_column}' not found in data"

    if split_method == "date":
        assert date_column in data.columns, "date_column is required for split_method='date'"

    # Check if Quarto CLI is installed
    if shutil.which("quarto") is None:
        raise EnvironmentError("Quarto CLI is not installed. Please install it from https://quarto.org")

    os.makedirs(output_dir, exist_ok=True)
    exp_id = str(uuid.uuid4())[:8]

    df = data.copy()

    # Remove rows with missing values
    null_rows = df.isnull().sum().sum()
    df = df.dropna()

    removed_rows = int(null_rows)

    # Drop datetime features
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns:
        if col != date_column:
            df = df.drop(columns=[col])

    # Encode categorical features
    le_dict = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

    # Train-test split
    if split_method == "date":
        df = df.sort_values(by=date_column)
        split_idx = int((1 - test_size) * len(df))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    models = []

    if task == "regression":
        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=100)),
            ("XGBRegressor", XGBRegressor(n_estimators=100, verbosity=0)),
            ("CatBoostRegressor", CatBoostRegressor(verbose=0)),
            ("LGBMRegressor", LGBMRegressor())
        ]
    else:
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=100)),
            ("XGBClassifier", XGBClassifier(n_estimators=100, verbosity=0)),
            ("CatBoostClassifier", CatBoostClassifier(verbose=0)),
            ("LGBMClassifier", LGBMClassifier())
        ]

    results = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if task == "regression":
            metrics = {
                "rmse_train": mean_squared_error(y_train, y_train_pred),
                "mae_train": mean_absolute_error(y_train, y_train_pred),
                "r2_train": r2_score(y_train, y_train_pred),
                "rmse_test": mean_squared_error(y_test, y_test_pred),
                "mae_test": mean_absolute_error(y_test, y_test_pred),
                "r2_test": r2_score(y_test, y_test_pred),
            }
        else:
            metrics = {
                "accuracy_train": accuracy_score(y_train, y_train_pred),
                "precision_train": precision_score(y_train, y_train_pred, average="macro"),
                "recall_train": recall_score(y_train, y_train_pred, average="macro"),
                "f1_train": f1_score(y_train, y_train_pred, average="macro"),
                "accuracy_test": accuracy_score(y_test, y_test_pred),
                "precision_test": precision_score(y_test, y_test_pred, average="macro"),
                "recall_test": recall_score(y_test, y_test_pred, average="macro"),
                "f1_test": f1_score(y_test, y_test_pred, average="macro"),
            }

        results.append({
            "model": name,
            "params": model.get_params(),
            **metrics
        })

    report_path = os.path.join(output_dir, f"report_{exp_id}.qmd")
    html_output = os.path.join(output_dir, f"report_{exp_id}.html")

    generate_quarto_report(
        report_path,
        results,
        task=task,
        split_method=split_method,
        test_size=test_size,
        removed_rows=removed_rows
    )

    subprocess.run(["quarto", "render", report_path, "--output", html_output])
    print(f"✅ Report generated: {html_output}")

def generate_quarto_report(path, results, task, split_method, test_size, removed_rows):
    with open(path, "w") as f:
                f.write(f"""---
        title: "Hasil Eksperimen ML"
        format: html
        editor: visual
        ---

        # 📊 Ringkasan Eksperimen

        - **Jenis Task**: {task}
        - **Metode Split**: {split_method}
        - **Proporsi Data Test**: {test_size}
        - **Baris yang Dihapus karena Null**: {removed_rows}

        # 🔍 Hasil Model

        """)
                for res in results:
                    f.write(f"## Model: {res['model']}\n")
                    f.write("### 🔧 Parameter\n")
                    f.write("```\n")
                    f.write(str(res["params"]))
                    f.write("\n```\n")
                    f.write("### 📈 Metrics\n")
                    f.write("| Metric | Value |\n|--------|--------|\n")
                    for k, v in res.items():
                        if k not in ["model", "params"]:
                            f.write(f"| {k} | {v:.4f} |\n")
                    f.write("\n---\n")

