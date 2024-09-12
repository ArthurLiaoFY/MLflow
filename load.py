import os
from configparser import ConfigParser

import mlflow

config = ConfigParser()
config.read("config.ini")

# Set MLflow tracking URI and environment variables
mlflow.set_tracking_uri(uri=config["tracking_server_url"])
os.environ["MLFLOW_TRACKING_USERNAME"] = config["tracking_username"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = config["tracking_password"]


def download_best_model(run_id):
    try:
        # Download the best model if found
        model_artifact_path = (
            "model"  # Assuming the model is stored under the artifact path "model"
        )
        mlflow.download_artifacts(run_id=run_id, artifact_path=model_artifact_path)
        print(f"最佳模型已下載至 {model_artifact_path}")
    except Exception as e:
        print(f"下載模型時出錯: {e}")


if __name__ == "__main__":
    # Hardcoded run_id for testing
    run_id = "5079ee18f904469f82453860429ac784"
    download_best_model(run_id)
