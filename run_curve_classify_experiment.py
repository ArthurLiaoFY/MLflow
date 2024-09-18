# %%
from configparser import ConfigParser

import torch
from sklearn.model_selection import train_test_split

from models.deep_models.models.conv_gru_att import Convolutional1DGRUAttention
from models.deep_models.utils.prepare_data import to_dataloader
from models.deep_models.utils.tools import (
    check_model_device,
    check_tensor_device,
    get_device,
)
from projects.curve_classify.load_data import load_data

# %%
device = get_device()
config = ConfigParser()
config.read("projects/curve_classify/curve_classify.ini")
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
train_loader = to_dataloader(
    cum_curve.iloc[train_idx, :].to_numpy(),
    label.loc[train_idx, "test_result"].to_numpy(),
)
valid_loader = to_dataloader(
    cum_curve.iloc[val_idx, :].to_numpy(),
    label.loc[val_idx, "test_result"].to_numpy(),
)
# %%
device = get_device()
conv_gru_att = Convolutional1DGRUAttention(
    conv_in_channels=1,  # C in shape : (B, C, H)
    conv_out_channels=32,  # C in shape : (B, C, H)
    gru_input_size=cum_curve.shape[1],
    gru_hidden_size=64,
    gru_layer_amount=2,
    attention_num_of_head=8,
    out_feature_size=2,
).to(device=device)
# %%

# %%
