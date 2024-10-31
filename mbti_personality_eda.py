# %%
from configparser import ConfigParser

from datasets import Features, Value, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from models.deep_models.utils.tools import get_device
from projects.mbti_personality.model import LanguageModel

# %%
device = get_device()
config = ConfigParser()
config.read("projects/mbti_personality/mbti_personality.ini")

tokenizer = AutoTokenizer.from_pretrained(config["model"].get("checkpoint"))
model = LanguageModel(checkpoint=config["model"]["checkpoint"], num_labels=16).to(
    device
)
# %%
train_dataset = load_dataset(
    path=config["path"]["data_file_path"],
    data_files={
        "train": "./mbti_1.csv",
    },
    features=Features(
        {
            "type": Value(dtype="string"),
            "posts": Value(dtype="string"),
        }
    ),
)

train_dataset = train_dataset.rename_column("type", "labels")
train_dataset = train_dataset.class_encode_column("labels")
train_dataset = train_dataset["train"].train_test_split(
    test_size=0.3,
    stratify_by_column="labels",
    seed=int(config["model"]["seed"]),
)
# %%
train_dataset = train_dataset.map(
    lambda dataset: tokenizer(
        dataset["posts"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    ),
    batched=True,
    remove_columns=["posts"],
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
optimizer = AdamW(model.parameters(), lr=3e-5)
num_training_steps = int(config["model"]["epoch"]) * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
import torch

from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import (
    Accuracy,
    AreaUnderCurve,
    Precision,
    Recall,
)
from models.deep_models.training.loss import cross_entropy_loss
from models.deep_models.training.train_model import finetune_llm_model

finetune_llm_model(
    run_id="test_llm",
    llm_model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=eval_dataloader,
    loss_fn=cross_entropy_loss,
    evaluate_fns={
        "Accuracy": Accuracy(),
        # "Precision": Precision(),
        # "Recall": Recall(),
        # "AUC": AreaUnderCurve(),
    },
    optimizer=torch.optim.Adam(model.parameters(), lr=float(1e-3)),
    early_stopping=EarlyStopping(
        patience=int(config["model"].get("patience")),
    ),
    log_file_path=config["path"].get("project_location"),
    mlflow_tracking=False,
)

# %%
