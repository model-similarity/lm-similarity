import datasets
import argparse
import sys
import os
import numpy as np
import yaml
from huggingface_hub import login
from lmsim.metrics import K_p

#define the path to the leaderboard
EVAL_ROOT = "open-llm-leaderboard/"
BENCH_ROOT = "leaderboard_mmlu_pro"

def load_settings(settings_file):
    # Get the directory one level up
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # Construct the file path
    file_path = os.path.join(parent_dir, settings_file)
    # Load the settings file
    with open(file_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

def softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum(axis=0)

def _load_data(model_name: str):

    model_name = model_name.replace("/", "__")
    # Load the dataset from hugingface hub                                 
    try:
        data =  datasets.load_dataset(
                    EVAL_ROOT+model_name+"-details",
                    name=model_name+ '__' + BENCH_ROOT,
                    split="latest")
        data = data.sort("doc_id")
        data = data.to_dict()
        return data
    
    except Exception as e:
        print(f'Failed to load data for {model_name} with error: {e}')
        sys.exit()


def main(args: argparse.Namespace):
    # Load the data
    model_a_data = _load_data(args.model_a)
    model_b_data = _load_data(args.model_b)

    # Assert that the documents are sorted in the same order
    assert model_a_data['doc_id'] == model_b_data['doc_id'], "Documents are not sorted in the same order"

    # Format to softmax output probabilities
    sofmax_a, softmax_b = [], []
    for sample_a, sample_b in zip(model_a_data['filtered_resps'], model_b_data['filtered_resps']):
        logits_a = np.array([float(r1[0]) for r1 in sample_a])
        logits_b = np.array([float(r2[0]) for r2 in sample_b])
        sofmax_a.append(softmax(logits_a))
        softmax_b.append(softmax(logits_b))
    
    #extract the ground truth index
    gt_a, gt_b = [], []
    for sample_a, sample_b in zip(model_a_data["doc"], model_b_data["doc"]):
        gt_a.append(int(sample_a["answer_index"]))
        gt_b.append(int(sample_b["answer_index"]))
    
    #assert that the ground truth index are the same
    if gt_a != gt_b:
        print("Ground truth index are not the same")
        print("Removing samples that mismatch:")
        indices_to_remove = []
        for idx, (a, b) in enumerate(zip(gt_a, gt_b)):
            if a != b:
                print(f"Removing sample {idx}")
                indices_to_remove.append(idx)
        for idx in sorted(indices_to_remove, reverse=True):
            del sofmax_a[idx]
            del softmax_b[idx]
            del gt_a[idx]
            del gt_b[idx]
    
    #assert that the all inputs are the same length
    assert len(sofmax_a) == len(softmax_b) ==  len(gt_a) == len(gt_b), "Inputs are not the same length"

    #compute the probabilistic error consistency
    kp = K_p()
    similarity = kp.compute_kp(sofmax_a, softmax_b, gt_a)
    print("=============RESULTS================")
    print(f"Similarity: {similarity:.2f}")
    print(f'Observed agreement: {kp.observed:.2f}')
    print(f'Expected agreement: {kp.expected:.2f}')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute probabilistic error consistency between model pair")
    parser.add_argument("--model_a", type=str, default="Qwen/Qwen2.5-72B-Instruct",help="Huggingface model name")
    parser.add_argument("--model_b", type=str, default="meta-llama/Llama-3.3-70B-Instruct",help="Huggingface model name")
    args = parser.parse_args()

    # Login to the Hugging Face Hub
    settings = load_settings("settings.yaml")
    hf_token = settings["huggingface"]["api_token"]
    login(token=hf_token)

    print(f'Computing probabilistic error consistency between models: {args.model_a} and {args.model_b}')

    main(args)