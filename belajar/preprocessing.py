# Preprocessing utilities
import pandas as pd

def clean_data(X, y):
    # Drop rows with missing values
    df = X.copy()
    df['__target__'] = y
    df.dropna(inplace=True)

    y_clean = df['__target__']
    X_clean = df.drop(columns='__target__')

    # Remove datetime features (only used for split)
    datetime_cols = X_clean.select_dtypes(include='datetime').columns.tolist()
    
    return X_clean, y_clean, datetime_cols
