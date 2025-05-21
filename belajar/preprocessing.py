# preprocessing.py
import pandas as pd

def clean_data(X, y):
    # Gabungkan X dan y untuk proses dropna yang konsisten
    df = X.copy()
    df['__target__'] = y
    df.dropna(inplace=True)

    # Pisahkan kembali X dan y
    y_clean = df['__target__']
    X_clean = df.drop(columns='__target__')

    # Identifikasi kolom datetime
    datetime_cols = X_clean.select_dtypes(include='datetime').columns.tolist()

    # Drop kolom datetime (tidak dipakai untuk modeling)
    X_clean = X_clean.drop(columns=datetime_cols)

    return X_clean, y_clean, datetime_cols
