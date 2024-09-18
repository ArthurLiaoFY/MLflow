import os

import mlflow


def setup_mlflow(mlflow_config):

    # Set MLflow tracking URI and environment variables
    mlflow.set_tracking_uri(uri=mlflow_config["MLflow"]["tracking_server_url"])
    if mlflow_config["MLflow"].get("tracking_username"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["MLflow"][
            "tracking_username"
        ]
    if mlflow_config["MLflow"].get("tracking_password"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["MLflow"][
            "tracking_password"
        ]


def setup_experiment(config):
    experiment = mlflow.get_experiment_by_name(config["experiment"]["experiment_name"])
    experiment_id = (
        mlflow.create_experiment(config["experiment"]["experiment_name"])
        if not experiment
        else (
            experiment.experiment_id
            if experiment.lifecycle_stage == "active"
            else experiment
        )
    )
    return experiment_id
