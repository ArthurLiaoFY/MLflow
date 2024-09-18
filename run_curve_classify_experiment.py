# %%
import os
from configparser import ConfigParser
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split

import mlflow
from models.deep_models.models.conv_gru_att import ConvolutionalGRUAttention
from models.deep_models.training.train_model import train_model
from models.deep_models.utils.prepare_data import to_dataloader
from models.deep_models.utils.tools import (
    check_model_device,
    check_tensor_device,
    get_device,
)
from projects.curve_classify.load_data import load_data
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
# %%
curve, cum_curve, label = load_data(
    data_file_path=config["curve_classify"]["data_file_path"]
)
train_idx, val_idx = train_test_split(
    label.index,
    train_size=0.8,
    random_state=1122,
    shuffle=True,
    stratify=label["test_result"],
)
# %%

# %%
conv_gru_att = ConvolutionalGRUAttention(
    conv_in_channels=1,  # C in shape : (B, C, H)
    conv_out_channels=32,  # C' in shape : (B, C', H)
    gru_input_size=cum_curve.shape[1],
    gru_hidden_size=64,
    gru_layer_amount=2,
    attention_num_of_head=8,
    out_feature_size=2,
).to(device=device)
import numpy as np

# %%
from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import RSquare
from models.deep_models.training.loss import root_mean_square_error

trained_nn_model = train_model(
    nn_model=conv_gru_att,
    train_dataloader=to_dataloader(
        cum_curve.iloc[train_idx, :].to_numpy()[:, np.newaxis, :],
        label.loc[train_idx, "test_result"].to_numpy()[:, np.newaxis],
        shuffle=True,
    ),
    valid_dataloader=to_dataloader(
        cum_curve.iloc[val_idx, :].to_numpy()[:, np.newaxis, :],
        label.loc[val_idx, "test_result"].to_numpy()[:, np.newaxis],
        shuffle=False,
    ),
    loss_fn=root_mean_square_error,
    evaluate_fns={"R-square": RSquare()},
    optimizer=torch.optim.Adam(conv_gru_att.parameters(), lr=0.005),
    early_stopping=EarlyStopping(patience=20),
    epochs=1000,
)
# %%
