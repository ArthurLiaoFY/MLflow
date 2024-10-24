from configparser import ConfigParser

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

config = ConfigParser()
config.read("projects/news_popularity/news_popularity.ini")

tokenizer = AutoTokenizer.from_pretrained(config["model"].get("checkpoint"))


def html_ingredients_extract(dataset):
    html_ingredients_list = []
    for idx in range(len(dataset["Page content"])):
        soup = BeautifulSoup(dataset["Page content"][idx], "html.parser")

        title = (
            soup.find("h1", class_="title").get_text(strip=True)
            if soup.find("h1", class_="title")
            else "N/A"
        )

        article_content = ""
        for paragraph in soup.find_all("p"):
            article_content += paragraph.get_text(strip=True) + "\n"

        html_ingredients_list.append("; ".join([title, article_content]))

    dataset["Page content"] = html_ingredients_list

    return tokenizer(
        dataset["Page content"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
