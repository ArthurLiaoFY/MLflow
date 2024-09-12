import os
import pickle
from configparser import ConfigParser
from datetime import datetime

import mlflow
from SCHEDULED_MODEL.Holmes.P.L5.train_L5_model import UphTrainPredictModel

config = ConfigParser()
config.read("config.ini")

# Set MLflow tracking URI and environment variables
mlflow.set_tracking_uri(uri=config["MLflow"]["tracking_server_url"])

if config["MLflow"].get("tracking_username"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = config["MLflow"]["tracking_username"]
if config["MLflow"].get("tracking_password"):
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config["MLflow"]["tracking_password"]


# Create an experiment and log two runs under it

experiment = mlflow.get_experiment_by_name(config["Holmes_L5"]["experiment_name"])
experiment_id = (
    mlflow.create_experiment(config["Holmes_L5"]["experiment_name"])
    if not experiment
    else (
        experiment.experiment_id
        if experiment.lifecycle_stage == "active"
        else experiment
    )
)

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="".join(
        [
            str(datetime.now().year),
            str(datetime.now().month),
            str(datetime.now().day),
            str(datetime.now().hour),
            str(datetime.now().minute),
            str(datetime.now().second),
        ]
    ),
) as run:

    #######
    # currently use pickle load
    # eventually, the data will be provided by backend API
    kwargs = {
        "model_file_path": "./SCHEDULED_MODEL/Holmes/P/L5/train_model_results",
        "model_type": "NN",
        "seed": 1122,
        "L5_early_stopping_patience": 20,
        "L5_learning_rate": 0.0005,
        "L5_epoch": 5,
        "db_address": "10.146.208.99.5432",
        "db_name": "WJ_C02",
        "owner_id": "1",
        "line_id": "1",
        "sector_id": "1",
        "run_id": run.info.run_id,
    }
    with open("agg_status_dict.pkl", "rb") as f:
        agg_status_dict = pickle.load(f)
    with open("prod_capacity_dict.pkl", "rb") as f:
        prod_capacity_dict = pickle.load(f)
    with open("sn_dict.pkl", "rb") as f:
        sn_dict = pickle.load(f)
    print("data collected")
    #######
    sim_model = UphTrainPredictModel(**kwargs)
    sim_model.train_model(
        agg_status_dict=agg_status_dict,
        prod_capacity_dict=prod_capacity_dict,
        sn_dict=sn_dict,
    )

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(kwargs)
