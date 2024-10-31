from datasets import Features, Value, load_dataset


def load_data(config):
    train_dataset = load_dataset(
        path=config["path"].get("data_file_path"),
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

    return train_dataset["train"].train_test_split(
        test_size=0.3,
        stratify_by_column="labels",
        seed=int(config["model"].get("seed")),
    )

