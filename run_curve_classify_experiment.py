# %%
from configparser import ConfigParser
from datetime import datetime

import mlflow
from models.deep_models.utils.tools import get_device
from projects.curve_classify.load_data import load_data
from projects.curve_classify.train import CurveClassify
from setup_mlflow import setup_experiment, setup_mlflow

device = get_device()

# %%
# setting up mlflow
mlflow_config = ConfigParser()
mlflow_config.read("mlflow_config.ini")
setup_mlflow(mlflow_config=mlflow_config)

config = ConfigParser()
config.read("projects/curve_classify/curve_classify.ini")
experiment_id = setup_experiment(config=config)

# with mlflow.start_run(
#     experiment_id=experiment_id,
#     run_name="".join(
#         [
#             str(datetime.now().year),
#             str(datetime.now().month),
#             str(datetime.now().day),
#             str(datetime.now().hour),
#             str(datetime.now().minute),
#             str(datetime.now().second),
#         ]
#     ),
# ) as run:
#     pass


with mlflow.start_run(
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

    curve, cum_curve, label = load_data(
        data_file_path=config["curve_classify"]["data_file_path"]
    )

    cc = CurveClassify(run_id="", **config["model"])
    cc.train_model(
        curve_array=cum_curve.to_numpy(),
        label_array=label["test_result"].to_numpy(),
    )
# TODO: test
# FIXME: test
