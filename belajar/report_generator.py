# Report generation logic
import subprocess
from pathlib import Path

def generate_report(X, y, results_df, task, test_size, split_method, total_rows, train_rows, test_rows):
    X.to_csv("reports/X.csv", index=False)
    y.to_csv("reports/y.csv", index=False)
    results_df.to_csv("reports/experiment_results.csv", index=False)

    with open("reports/config.txt", "w") as f:
        f.write(f"task={task}\n")
        f.write(f"split_method={split_method}\n")
        f.write(f"test_size={test_size}\n")
        f.write(f"rows_total={total_rows}\n")
        f.write(f"rows_train={train_rows}\n")
        f.write(f"rows_test={test_rows}\n")
        f.write(f"missing_value_handling=drop_rows\n")

    subprocess.run(["quarto", "render", "reports/report_template.qmd"], check=True)
