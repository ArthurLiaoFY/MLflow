from configparser import ConfigParser

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

config = ConfigParser()
config.read("projects/mbti_personality/mbti_personality.ini")

tokenizer = AutoTokenizer.from_pretrained(config["model"].get("checkpoint"))


def html_ingredients_extract(dataset):

    return tokenizer(
        dataset["Page content"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
