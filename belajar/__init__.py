"""
belajar - Library otomatisasi eksperimen machine learning untuk regresi dan klasifikasi.

Fitur:
- Mendukung berbagai model: Linear/Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM
- Preprocessing otomatis: drop kolom datetime, encoding kategori, drop missing value
- Split data berdasarkan waktu atau acak
- Evaluasi metrik otomatis dan pelaporan via Quarto
"""

__version__ = "0.1.2"

from .core import run_experiment

