import os
import json
import re
import argparse

from pathlib import Path


def filter_response(response):
    """
    Truncate the response at the end of the first occurrence of 'answer is (XYZ)'.
    If the pattern is not found, return the full response.
    
    Args:
        response (str): The response string to process.
    
    Returns:
        str: The truncated or original response.
    """
    # Regular expression to match "answer is (XYZ)"
    pattern = r"answer is \([^)]+?\)"
    
    # Find the first match
    match = re.search(pattern, response)
    if match:
        # Truncate the response at the end of the first match
        return response[:match.end()]
    return response


def filter_and_merge_samples(input_dir, output_dir, filter_path):
    """
    Filters and merges sample files from multiple model directories.
    This function reads sample files from subdirectories within the input directory,
    filters the samples based on question IDs specified in a filter file, and merges
    the filtered samples into a single JSON file for each model. Additionally, it copies
    "results" files from the input directory to the output directory.
    Args:
        input_dir (str): The path to the input directory containing model subdirectories.
        output_dir (str): The path to the output directory where filtered samples and results will be saved.
        filter_path (str): The path to the filter file containing question IDs to filter by.
    Raises:
        FileNotFoundError: If the filter file or any sample file is not found.
        json.JSONDecodeError: If there is an error decoding a JSON sample file.
    Example:
        filter_and_merge_samples('/path/to/input', '/path/to/output', '/path/to/filter.txt')
    """
    # Load question IDs from filter file
    with open(filter_path, 'r') as f:
        filter_ids = set(int(line.strip()) for line in f if line.strip())

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over each subdirectory (model directory) in the input directory
    for model_dir in os.listdir(input_dir):
        model_path = os.path.join(input_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Copy "results" file to output directory
        for file_name in os.listdir(model_path):
            if file_name.startswith("results"):
                src_file = os.path.join(model_path, file_name)
                dst_file = os.path.join(output_dir, model_dir, file_name)
                Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

        # Combine "samples" files and filter them
        combined_samples = []
        for file_name in os.listdir(model_path):
            if file_name.startswith("samples"):
                file_path = os.path.join(model_path, file_name)
                with open(file_path, 'r') as f:
                    print(f"Reading samples from {file_path}")
                    for line in f:
                        sample = json.loads(line)
                        doc = sample.get("doc", {})
                        question_id = doc.get("question_id")
                        if question_id in filter_ids:
                            combined_sample = {
                                "question_id": doc.get("question_id"),
                                "question": doc.get("question"),
                                "options": doc.get("options"),
                                "answer": doc.get("answer"),
                                "answer_index": doc.get("answer_index"),
                                "target": sample.get("target"),
                                "filtered_resps": sample.get("filtered_resps")[0],
                                "resps": filter_response(sample.get("resps")[0][0]),
                                "category": doc.get("category"),
                            }
                            combined_samples.append(combined_sample)

        # Sort samples by question_id
        combined_samples.sort(key=lambda x: x.get("question_id"))

        # Print number of samples
        print(f"Model: {model_dir}, Number of samples: {len(combined_samples)}")

        # Save combined and filtered samples to a single JSON file
        if combined_samples:
            output_file = os.path.join(output_dir, model_dir, "samples.json")
            with open(output_file, 'w') as f:
                json.dump(combined_samples, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing model subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory where filtered samples and results will be saved.")
    parser.add_argument("--filter_path", type=str, required=True, help="Path to the filter file containing question IDs to filter by.")

    args = parser.parse_args()

    filter_and_merge_samples(args.input_dir, args.output_dir, args.filter_path)