import os
import pickle
from configparser import ConfigParser
from datetime import datetime, timedelta

import pandas as pd

import mlflow

config = ConfigParser()
config.read("config.ini")

# Set MLflow tracking URI and environment variables
mlflow.set_tracking_uri(uri=config["tracking_server_url"])
os.environ["MLFLOW_TRACKING_USERNAME"] = config["tracking_username"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = config["tracking_password"]

# 获取实验ID
experiment_id = "Holmes_P_L3_10.146.208.117.5432_holmes_1_1_1_1_1"

# 搜索实验中的运行记录
experiment_detail_df = mlflow.search_runs(experiment_ids=[experiment_id])

# 过滤运行记录，选取最近完成的模型和标记为最佳的模型
recent_runs = experiment_detail_df[
    (experiment_detail_df["status"] == "FINISHED")
    & (
        pd.to_datetime(experiment_detail_df["start_time"])
        >= datetime.today() - timedelta(days=60)
    )
]

best_model_runs = experiment_detail_df[
    (experiment_detail_df["status"] == "FINISHED")
    & (
        pd.to_datetime(experiment_detail_df["start_time"])
        >= datetime.today() - timedelta(days=7)
    )
    & (experiment_detail_df["tags.best_model"] == "True")
]

# 打印所有模型的相关信息
print("Recent finished runs:")
print(recent_runs[["run_id", "status", "start_time", "end_time", "tags.best_model"]])

print("\nBest model runs:")
print(
    best_model_runs[["run_id", "status", "start_time", "end_time", "tags.best_model"]]
)
