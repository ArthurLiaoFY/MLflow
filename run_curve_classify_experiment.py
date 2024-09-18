# %%
from configparser import ConfigParser

import torch
from sklearn.model_selection import train_test_split

from models.deep_models.models.conv_gru_att import Convolutional1DGRUAttention
from models.deep_models.utils.prepare_data import to_dataloader
from models.deep_models.utils.tools import check_model_device, get_device
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
train_x = cum_curve.iloc[train_idx, :]
val_x = cum_curve.iloc[val_idx, :]
train_y = label.iloc[train_idx, :]
val_y = label.iloc[val_idx, :]
# %%
device = get_device()
conv_gru_att = Convolutional1DGRUAttention(
    conv_in_channels=train_x.shape[1],
    conv_out_channels=128,
    gru_input_size=128,
    gru_hidden_size=128,
    gru_layer_amount=2,
    out_feature_size=2,
).to(device=device)
# %%

x = torch.randn(size=(64, train_x.shape[1])).to(device=device)
x.shape
# %%
check_model_device(model=conv_gru_att)
# %%
conv_gru_att.forward(input_tensor=x)
# %%
