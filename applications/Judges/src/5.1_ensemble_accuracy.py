import json
import os 
import numpy as np  
import csv
import argparse
    

def get_ensemble_decision(sample_dict):
    """
    Determines the ensemble decision based on the votes from multiple models.
    Args:
        sample_dict (dict): A dictionary where keys are strings that include "score_" 
                            and values are "1" for correct predictions and "0" for incorrect predictions.
    Returns:
        int: Returns 1 if the number of correct votes is greater than the number of incorrect votes,
             otherwise returns 0.
    """

    votes = {"correct": 0, "incorrect": 0}
    for key, value in sample_dict.items():
        if "score_" in key:
            if value == "1":
                votes["correct"] += 1
            elif value == "0":
                votes["incorrect"] += 1
    return 1 if votes["correct"] > votes["incorrect"] else 0


def get_ensemble_accuracy(data_dir):
    """
    Calculate and save the ensemble accuracy for models in a given directory.
    This function iterates over each subdirectory (representing a model) in the specified
    input directory, combines the "samples" files, filters them, and calculates the ensemble
    accuracy for each model. The results are saved to a CSV file in the input directory.
    Args:
        data_dir (str): The path to the directory containing model subdirectories.
    Returns:
        None
    Raises:
        FileNotFoundError: If the "samples.json" file is not found in a model directory.
        json.JSONDecodeError: If there is an error decoding the JSON file.
        IOError: If there is an error reading or writing files.
    """

    # Iterate over each subdirectory (model directory) in the input directory
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    ensemble_accuracies = []
    for model_dir in dir_list:
        model_path = os.path.join(data_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Combine "samples" files and filter them
        ensemble_judgements = []
        samples_path = os.path.join(model_path, "samples.json")
        with open(samples_path, 'r') as f:
            samples = json.load(f)
            for sample in samples:
                ensemble_judgements.append(get_ensemble_decision(sample))

        # Calculate ensemble accuracy
        n_correct = np.sum(np.array(ensemble_judgements))
        n_total = len(ensemble_judgements)
        accuracy = n_correct / n_total
        print(f"Model: {model_dir}, Accuracy: {accuracy:.3f}")

        model_entry = {"model": model_dir, "accuracy": accuracy}
        ensemble_accuracies.append(model_entry)

    # Save ensemble accuracies to output directory
    keys = ensemble_accuracies[0].keys()
    output_path = os.path.join(data_dir, "ref_opt_ensemble_acc.csv")
    with open(output_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(ensemble_accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ensemble accuracy for models in a directory.")
    parser.add_argument("data_dir", type=str, help="The path to the directory containing model subdirectories.")
    args = parser.parse_args()

    get_ensemble_accuracy(args.data_dir)