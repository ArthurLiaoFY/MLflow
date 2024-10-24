# %%
from configparser import ConfigParser

from datasets import Features, Value, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from projects.news_popularity.model import LanguageModel
from projects.news_popularity.preprocess import html_ingredients_extract, tokenizer
from utils.device import device

config = ConfigParser()
config.read("projects/news_popularity/news_popularity.ini")


model = LanguageModel(checkpoint=config["model"]["checkpoint"]).to(device)
# %%
train_dataset = load_dataset(
    path=config["path"]["data_file_path"],
    data_files={
        "train": "./train.csv",
    },
    features=Features(
        {
            "Id": Value(dtype="string"),
            "Page content": Value(dtype="string"),
            "Popularity": Value(dtype="string"),
        }
    ),
).select_columns(["Id", "Page content", "Popularity"])

train_dataset = train_dataset.rename_column("Popularity", "labels")
train_dataset = train_dataset.class_encode_column("labels")
train_dataset = train_dataset["train"].train_test_split(
    test_size=0.3,
    stratify_by_column="labels",
    seed=int(config["model"]["seed"]),
)
train_dataset = train_dataset.map(
    html_ingredients_extract,
    batched=True,
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
    print(
        model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
    )
    break

# %%
