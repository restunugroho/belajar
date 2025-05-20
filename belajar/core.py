import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tqdm import tqdm

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def run_experiment(
    data: pd.DataFrame,
    target: str,
    task: str = "regression",
    split_method: str = "date",
    split_column: str = None,
    test_size: float = 0.2,
    date_threshold: str = None,
    output_dir: str = "output"
):
    assert task in ["regression", "classification"], "Invalid task"
    assert split_method in ["random", "date"], "Invalid split method"
    assert target in data.columns, "Target not found in data"

    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()

    # Drop rows with missing values
    df = df.dropna()
    dropped = data.shape[0] - df.shape[0]

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        if col != target:
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=[target])
    y = df[target]

    if split_method == "random":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    elif split_method == "date":
        assert split_column is not None, "Date column must be provided for date-based split"
        df[split_column] = pd.to_datetime(df[split_column])
        df = df.sort_values(by=split_column)
        split_idx = int((1 - test_size) * len(df))
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        X_test = test_data.drop(columns=[target])
        y_test = test_data[target]

    model_dict = {
        "linear": LinearRegression() if task == "regression" else LogisticRegression(max_iter=1000),
        "rf": RandomForestRegressor() if task == "regression" else RandomForestClassifier(),
        "xgb": XGBRegressor() if task == "regression" else XGBClassifier(),
        "cat": CatBoostRegressor(verbose=0) if task == "regression" else CatBoostClassifier(verbose=0),
        "lgbm": LGBMRegressor() if task == "regression" else LGBMClassifier()
    }

    results = []

    for name, model in tqdm(model_dict.items(), desc="Training models"):
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
            "test_metrics": test_score
        })

        # SHAP Analysis
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

    report_path = os.path.join(output_dir, "report.qmd")
    with open(report_path, "w") as f:
        f.write("# Hasil Eksperimen\n\n")
        f.write(f"Dropped missing rows: {dropped}\n\n")
        f.write(f"Train/Test split method: {split_method}, test size: {test_size}\n\n")
        for res in results:
            f.write(f"## Model: {res['model']}\n")
            f.write("### Train Metrics:\n")
            for k, v in res["train_metrics"].items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write("### Test Metrics:\n")
            for k, v in res["test_metrics"].items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write(f"![SHAP Plot](shap_{res['model']}.png)\n\n")
