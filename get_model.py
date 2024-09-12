import os
from configparser import ConfigParser

import pandas as pd

import mlflow

config = ConfigParser()
config.read("config.ini")

# Set MLflow tracking URI and environment variables
mlflow.set_tracking_uri(uri=config["tracking_server_url"])
os.environ["MLFLOW_TRACKING_USERNAME"] = config["tracking_username"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = config["tracking_password"]

# Define experiment ID
experiment_id = 1  # Replace with the correct integer experiment ID

# Define evaluation metrics
metrics_to_consider = ["accuracy", "recall", "f1_score"]

# Search runs
experiment_detail_df = mlflow.search_runs(experiment_ids=[experiment_id])

# Filter runs based on status and recent timeframe
filtered_runs = experiment_detail_df[
    (experiment_detail_df["status"] == "FINISHED")
    & (
        pd.to_datetime(experiment_detail_df["start_time"])
        >= pd.Timestamp.today() - pd.Timedelta(days=60)
    )
]

# Compare run metrics and select the best model
best_run = None
best_score = None
for idx, run in filtered_runs.iterrows():
    metrics = mlflow.get_run(run_id=run["run_id"]).data.metrics
    current_score = sum([metrics[metric] for metric in metrics_to_consider])
    if best_run is None or current_score > best_score:
        best_run = run
        best_score = current_score

# Download the best model if found
if best_run is not None:
    model_artifact_path = (
        "model"  # Assuming the model is stored under the artifact path "model"
    )
    mlflow.download_artifacts(
        run_id=best_run["run_id"], artifact_path=model_artifact_path
    )
    print(f"最佳模型已下載至 {model_artifact_path}")
else:
    print("未找到符合條件的運行")
