import os
import pickle
from configparser import ConfigParser
from datetime import datetime

import mlflow
from SCHEDULED_MODEL.Holmes.P.L3.train_L3_model import UphTrainSimulationModel

config = ConfigParser()
config.read("config.ini")

# Set MLflow tracking URI and environment variables
mlflow.set_tracking_uri(uri=config["MLflow"]["tracking_server_url"])
os.environ["MLFLOW_TRACKING_USERNAME"] = config["MLflow"]["tracking_username"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = config["MLflow"]["tracking_password"]


# Create an experiment and log two runs under it

experiment = mlflow.get_experiment_by_name(config["Holmes_L3"]["experiment_name"])
experiment_id = (
    mlflow.create_experiment(config["Holmes_L3"]["experiment_name"])
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
        "model_file_path": "./SCHEDULED_MODEL/Holmes/P/L3/train_model_results",
        "model_type": "NN",
        "seed": 1122,
        "L3_early_stopping_patience": 20,
        "L3_validation_size": 0.3,
        "L3_learning_rate": 0.0005,
        "L3_epoch": 5,
        "db_address": "10.146.208.99.5432",
        "db_name": "WJ_C02",
        "owner_id": "1",
        "line_id": "1",
        "sector_id": "1",
        "run_id": run.info.run_id,
    }
    with open("masked_simulate_dict.pkl", "rb") as f:
        masked_simulate_dict = pickle.load(f)
    with open("masked_uph_dict.pkl", "rb") as f:
        masked_uph_dict = pickle.load(f)
    print("data collected")
    #######
    sim_model = UphTrainSimulationModel(**kwargs)
    sim_model.train_model(
        masked_simulate_dict=masked_simulate_dict,
        masked_uph_dict=masked_uph_dict,
    )

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(kwargs)
