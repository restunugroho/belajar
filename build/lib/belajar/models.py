# Model training and evaluation logic
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np

def train_models(X_train, y_train, X_test, y_test, task):
    results = []

    model_config = {
        'classification': {
            'LogisticRegression': (
                LogisticRegression(max_iter=1000),
                {'C': [0.1, 1, 10]}
            ),
            'RandomForestClassifier': (
                RandomForestClassifier(),
                {'n_estimators': [50, 100]}
            )
        },
        'regression': {
            'LinearRegression': (
                LinearRegression(),
                {}
            ),
            'RandomForestRegressor': (
                RandomForestRegressor(),
                {'n_estimators': [50, 100]}
            )
        }
    }

    for name, (model, param_grid) in model_config[task].items():
        search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy' if task == 'classification' else 'r2')
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluation
        metrics = get_metrics(y_train, y_train_pred, y_test, y_test_pred, task)

        results.append({
            'model': name,
            'best_params': search.best_params_,
            **metrics
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values('test_score', ascending=False).iloc[0]
    best_model_name = best_row['model']
    best_model = model_config[task][best_model_name][0].set_params(**best_row['best_params'])
    best_model.fit(X_train, y_train)

    return results_df, best_model

def get_metrics(y_train, y_train_pred, y_test, y_test_pred, task):
    if task == 'classification':
        return {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='macro'),
            'train_recall': recall_score(y_train, y_train_pred, average='macro'),
            'train_f1': f1_score(y_train, y_train_pred, average='macro'),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, average='macro'),
            'test_recall': recall_score(y_test, y_test_pred, average='macro'),
            'test_f1': f1_score(y_test, y_test_pred, average='macro'),
            'test_score': accuracy_score(y_test, y_test_pred),
        }
    else:
        return {
            'train_r2': r2_score(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_score': r2_score(y_test, y_test_pred),
        }
