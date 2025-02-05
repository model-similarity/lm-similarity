import json
import yaml
import os
import datasets
import pandas as pd
import numpy as np


def load_settings(settings_file):
    with open(settings_file, "r") as file:
        settings = yaml.safe_load(file)
    return settings


def exctract_indices(data: list):
    indices = [sample["question_id"] for sample in data]
    return indices


def load_json(filename: str):
    with open(filename, "r") as file:
        data = json.load(file)

    return exctract_indices(data)


def load_jsonl(data_path: str, model_name: str):
    for file in os.listdir(data_path):
        if (
            file.endswith("jsonl")
            and model_name == file.split("_leaderboard_mmlu_pro")[0]
        ):
            try:
                with open(os.path.join(data_path, file), "r") as f:
                    data = [json.loads(line) for line in f]
                data = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
                data = data.sort("doc_id")
                return data.to_dict()
            except:
                return "while loading manually from jsonl"
            
def softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum(axis=0)
