# %%
from configparser import ConfigParser

from datasets import Features, Value, load_dataset
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from models.deep_models.utils.tools import get_device
from projects.mbti_personality.model import LanguageModel

# %%
device = get_device()
config = ConfigParser()
config.read("./projects/mbti_personality/mbti_personality.ini")
secret_config = ConfigParser()
secret_config.read("./mlflow_config.ini")
# %%

login(token=secret_config["HuggingFace_Token"].get("token"))

model = LanguageModel(checkpoint="meta-llama/Llama-3.2-3B-Instruct").to(device)
# %%
train_dataset = load_dataset(
    path=config["path"]["data_file_path"],
    data_files={
        "train": "./en_decision_feeling.json",
    },
    field="data",
).select_columns(["instruction", "output"])
# %%
train_dataset = train_dataset["train"].train_test_split(
    test_size=0.3,
    seed=int(config["model"]["seed"]),
)
train_dataset.set_format("torch")
# %%
train_dataloader = DataLoader(
    train_dataset["train"],
    shuffle=True,
    batch_size=int(config["model"]["batch_size"]),
)
eval_dataloader = DataLoader(
    train_dataset["test"],
    batch_size=int(config["model"]["batch_size"]),
)
# %%
for batch in train_dataloader:
    print(batch)
    break


# %%
