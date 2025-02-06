import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import datasets
from huggingface_hub import login
from utils import load_settings, load_jsonl, softmax
from lmsim.metrics import CAPA, EC

# Define evalution plarform and benchmark root
EVAL_ROOT = "open-llm-leaderboard/"
BENCH_ROOT = "leaderboard_mmlu_pro"



class ModelMetrics:
    """
    A class to handle loading and processing of model evaluation data.
    Attributes:
    -----------
    name : str
        The name of the model.
    data : List[Dict]
        The evaluation data for the model.
    Methods:
    --------
    __init__(model_name: str, eval_name: str, dataset_name: str):
        Initializes the ModelMetrics instance and loads the data.
    _load_data(model_name: str, eval_name: str, dataset_name: str) -> List[Dict]:
        Loads the evaluation data for the specified model, evaluation, and dataset.
    """

    def __init__(self, model_name: str):
        self.name = model_name
        self.data = self._load_data(model_name)

    @staticmethod
    def _load_data(model_name: str) -> List[Dict]:
        def _load_data(
            model_name: str
        ) -> List[Dict]:
            """
            Load data for a given model, evaluation, and dataset.
            This function loads data either from a local JSONL file or from the Hugging Face Hub,
            depending on the specified model name.
            Args:
                model_name (str): The name of the model for which data is to be loaded.
                eval_name (str): The name of the evaluation.
                dataset_name (str): The name of the dataset.
            Returns:
                List[Dict]: A list of dictionaries containing the loaded data.
            Raises:
                Exception: If there is an error while loading data from the Hugging Face Hub.
            """

        # Manually loaded data for 3 models
        if model_name in [
            "google__gemma-2-2b-it",
            "google__gemma-2-9b",
            "google__gemma-2-9b-it",
        ]:
            data = load_jsonl(f"../data/{BENCH_ROOT}", model_name)
            return data

        # Load data from the hf hub
        elif model_name in [
            "meta-llama__Meta-Llama-3.1-8B",
            "meta-llama__Meta-Llama-3.1-70B",
        ]:
            size = model_name.split("-")[-1]
            try:
                data = datasets.load_dataset(
                    EVAL_ROOT + model_name + "-details",
                    name="llhf__3.1_Jul10-" + size + "__" + BENCH_ROOT,
                    split="latest",
                )
                data = data.sort("doc_id")
                data = data.to_dict()
            except Exception as e:
                data = "while loading from the hf hub via datasets.load_dataset"
        else:
            try:
                data = datasets.load_dataset(
                    EVAL_ROOT + model_name + "-details",
                    name=model_name + "__" + BENCH_ROOT,
                    split="latest",
                )
                data = data.sort("doc_id")
                data = data.to_dict()
            except Exception as e:
                data = "while loading from the hf hub via datasets.load_dataset"
        return data
    

def format_softmax_outputs(model1, model2, filter_mask, softmax):
    # Build a list of (softmax_a, softmax_b, gt_a, gt_b) in one pass
    results = [
        (
            softmax(np.fromiter((float(r1[0]) for r1 in sample_a_resp), dtype=float)),
            softmax(np.fromiter((float(r2[0]) for r2 in sample_b_resp), dtype=float)),
            int(sample_a_doc["answer_index"]),
            int(sample_b_doc["answer_index"])
        )
        for (sample_a_resp, sample_b_resp, sample_a_doc, sample_b_doc, keep) in zip(
            model1.data['filtered_resps'],
            model2.data['filtered_resps'],
            model1.data['doc'],
            model2.data['doc'],
            filter_mask
        )
        if keep
    ]
    
    # If no items passed the filter_mask, return empty lists
    if not results:
        return [], [], [], []
    
    # Unzip the list of tuples into four lists
    softmax_a, softmax_b, gt_a, gt_b = zip(*results)
    return list(softmax_a), list(softmax_b), list(gt_a), list(gt_b)



def calculate_and_save_metrics(
    model1: ModelMetrics,
    model2: ModelMetrics,
    filter_indices: List[int],
    diffpath: str,
    metrics: List[str],
) -> Dict:
    """
    Calculate several similarity metrics between two models and save the results to JSON files.
    Args:
        model1 (ModelMetrics): Metrics data for the first model.
        model2 (ModelMetrics): Metrics data for the second model.
        diffpath (str): The directory path where the results will be saved.
        metrics (List[str]): The list of metrics to calculate.
    Returns:
        Dict: A dictionary containing the calculated metrics and their results.
    The function performs the following steps:
    1. Creates the output directory based on the calculator's type and correlation flag.
    2. Calculates various metrics including cobs, kappa, Scott's Pi, Goel's score, and error consistency.
    3. Stores the results in a dictionary.
    4. Saves the results and any invalid indices to JSON files in the specified directory.
    """

    # Create output directory
    os.makedirs(diffpath, exist_ok=True)

    # Assert that the documents are sorted in the same order
    assert model1.data['doc_id'] == model2.data['doc_id'], "Documents are not sorted in the same order"

    # Create a filter mask based on the indices
    filter_mask = [(doc['question_id'] in filter_indices) for doc in model1.data['doc']]

    # Format to softmax output probabilities and extract gt 
    softmax_a, softmax_b, gt_a, gt_b = format_softmax_outputs(model1, model2, filter_mask, softmax)

    #assert that the ground truth index are the same
    indices_to_remove = []
    if gt_a != gt_b:
        for idx, (a, b) in enumerate(zip(gt_a, gt_b)):
            if a != b:
                indices_to_remove.append(idx)
        for idx in sorted(indices_to_remove, reverse=True):
            del softmax_a[idx]
            del softmax_b[idx]
            del gt_a[idx]
            del gt_b[idx]
    
    #assert that the all inputs are the same length
    assert len(softmax_a) == len(softmax_b) ==  len(gt_a) == len(gt_b), "Inputs are not the same length"

    results_dict = {"num_qs":len(softmax_a), "accuracy":""}
    for metric_name in metrics:
        if metric_name == 'capa':
            metric = CAPA()
        elif metric_name == 'capa_discrete':
            metric = CAPA(prob=False)
        elif metric_name == 'ec':
            metric = EC()
    
        similarity = metric.compute_k(softmax_a, softmax_b, gt_a)
   

        # Store results
        results_dict["accuracy"] =  {model1.name: metric.acc_model1, model2.name: metric.acc_model2}
        results_dict[metric_name] = {
                "similarity": similarity,
                "oberved": metric.observed,
                "expected": metric.expected,
                }
        if metric_name.find('capa') != -1:
            results_dict[metric_name]["p_hat"] = {model1.name: metric.p_hat_a, model2.name: metric.p_hat_b}
            results_dict[metric_name]["frac"] = metric.frac

            
    output_path = diffpath+ f"/similarity___{model1.name}___{model2.name}.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    if indices_to_remove:
        path = diffpath + f"/invalid_indices/"
        os.makedirs(path, exist_ok=True)
        output_path_invalid = (
            path + f"/invalid_indices___{model1.name}___{model2.name}.json"
        )
        with open(output_path_invalid, "w") as f:
            json.dump(indices_to_remove, f, indent=4)

    return results_dict


def process_model_pair(args: Tuple[str, str, str, str]) -> None:
    """
    Processes a pair of models by loading their metrics, filtering data, and calculating metrics.
    Args:
        args (Tuple[str, str, str, str, str, str, bool]): A tuple containing the following elements:
            - judge (str): The identifier for the judge model.
            - model (str): The identifier for the model to be compared.
            - diffpath (str): The directory path where differences and errors will be saved.
            - metrics (List[str]): The list of metrics to calculate.
    Returns:
        None
    """

    judge, model, diffpath, metrics = args

    # Load the correct indices
    with open("../data/filtered_question_ids.txt", "r") as file:
        filter_indices = set([int(line.strip()) for line in file])

    # load data
    model1 = ModelMetrics(judge)
    model2 = ModelMetrics(model)

    # if fail to load a model report the error
    if isinstance(model1.data, str) or isinstance(model2.data, str):
        if isinstance(model1.data, str):
            error_message = f"Error loading data for judge {judge} (in a pair with model {model}), error in function {model1.data}\n"
        elif isinstance(model2.data, str):
            error_message = f"Error loading data for model {model} (in pair with judge {judge}, error in function {model2.data})\n"

        # save the error message
        with open(f"{diffpath}/errors.txt", "a") as f:
            f.write(error_message)
            f.write("\n")
        return None

    else:
        # Calculate metrics and save results
        results = calculate_and_save_metrics(model1, model2, filter_indices, diffpath, metrics)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarity between model pairs"
    )
    parser.add_argument(
        "--metrics",                                           # Argument name (use it with --items in CLI)
        nargs="+",                                           # "+" means one or more items
        type=str,                                            # Parse each item as a string
        default=['capa','capa_discrete', 'ec'],              # Optional: if not provided, default to an empty list
        help="List of strings to process."
    )

    parser.add_argument(
        "--output_path", type=str, default="../output/sim", help="type of metric prob or not"
    )
    args = parser.parse_args()
    print(args)

    # Login to the Hugging Face Hub
    settings = load_settings("settings.yaml")
    hf_token = settings["huggingface"]["api_token"]
    login(token=hf_token)

    # define models
    models = (
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "microsoft/phi-4",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "tiiuae/Falcon3-1B-Instruct",
        "tiiuae/Falcon3-7B-Instruct",
        "tiiuae/Falcon3-10B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        # "HuggingFaceTB/SmolLM2-135M", - repeated crashes
        # "HuggingFaceTB/SmolLM2-360M", - not run on oLLMLv2
        "HuggingFaceTB/SmolLM2-1.7B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-70B",
        # "meta-llama/Llama-3.3-70B", - does not exist
        # "tiiuae/Falcon3-1B-Base", - unable to load weights
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "mistralai/Ministral-8B-Instruct-2410",
    )

    # Define judges
    judges = (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "mistralai/Ministral-8B-Instruct-2410",
    )

    
    # Prepare arguments for parallel processing
    process_args = []

    # Loop over judge and model pair to compute similarity
    for judge in judges:
        for model in models:
            # if judge != model:
            judge = judge.replace("/", "__")
            model = model.replace("/", "__")
            # Check if the file already exists
            path = f"{args.output_path}/similarity___{judge}___{model}.json"

            if os.path.exists(path):
                print(f"File {path} already exists")
                continue
            else:
                # print(f"Processing {judge} and {model}")
                process_args.append(
                    (
                        judge,
                        model,
                        args.output_path,
                        args.metrics
                    )
                )

    # Process model pairs in parallel
    with ProcessPoolExecutor() as executor:
        for _ in tqdm(
            executor.map(process_model_pair, process_args, chunksize=10),
            total=len(process_args),
            desc="Processing model pairs",
        ):
            pass

    print("Completed all model pairs")


if __name__ == "__main__":
    main()
