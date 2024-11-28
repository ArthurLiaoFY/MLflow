# %%
from configparser import ConfigParser

from projects.curve_classify.finetune import TFCFinetune
from projects.curve_classify.load_data import load_train_data
from projects.curve_classify.pretrain import TFCPretrain

# %%

config = ConfigParser()
config.read("projects/curve_classify/curve_classify.ini")
# %%
train_x_t, train_x_f, train_y = load_train_data(
    data_file_path=config["curve_classify"]["data_file_path"],
    augmentation=False,
)
# %%


# %%
pt = TFCPretrain(run_id=None, **config["model"])
# %%
ft = TFCFinetune(run_id=None, **config["model"])
# %%
ft.model
# %%
pt.model
# %%
