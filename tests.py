import belajar
import pandas as pd
from belajar import run_experiment
print(belajar.__version__)

df = pd.read_parquet('dataset.parquet')
print(df.shape)

features = ['request_time','gap_eta_traffic','eta_with_traffic','eta_without_traffic','est_distance','minutes','is_holiday',
            'pickup_distance_to_center','dropoff_distance_to_center',
            'req_pickup_longitude','req_pickup_latitude','req_dropoff_longitude','req_dropoff_latitude','kmph','argo_drop_waiting'
            ]

run_experiment(
    data=df[features],
    target="argo_drop_waiting",         # nama kolom target
    task="regression",             # atau "classification"
    split_method="random",           # atau "random"
#    date_column="request_time",         # kolom datetime untuk split
    test_size=0.3,
    output_dir="hasil_laporan"
)
