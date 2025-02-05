import os
import json
import argparse
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def get_prompt_wo_reference(question, response):
    prompt = f"""Your task is to judge whether the given response to a question is correct or not. You are only \
given a question and the response you are judging. 

Possible judgments:
"0": The response is incorrect.
"1": The response is correct.
    
Question: "{question}"
Response: "{response}"
    
To the best of your knowledge: Does the provided response answer the question correctly? This is part of an automated \
evaluation process, therefore you must only output a single word: "0" or "1". Do not justify your decision.
    
Evaluation (0/1):"""

    return prompt


def get_prompt_w_options(question, response, options, answer_index):
    options_wo_answer = options[:answer_index] + options[answer_index + 1:]
    prompt = f"""Your task is to judge whether the given response to a question is correct or not. You are given a \
question, a ground truth response, incorrect options and the response you are judging.

Possible judgments:
"0": The response is incorrect. It does not match the ground-truth answer or is more similar to any of the incorrect \
options than to the ground-truth answer.
"1": The response is correct. It matches the ground-truth.

Question: "{question}"
Ground truth: "{options[answer_index]}"
""" + "\n".join([f"Incorrect option ({i}): \"{opt}\"" for i, opt in enumerate(options_wo_answer)]) + f"""
Response: "{response}"

To the best of your knowledge: Does the provided response answer the question correctly, taking the ground-truth \
and wrong answer options into account? This is part of an automated evaluation process, therefore you must only \
output a single word: "0" or "1". Do not justify your decision.

Evaluate (0/1):"""
    return prompt

def get_judge_scores_wo_references(judge, tokenizer, system_prompt, questions, responses):
    """
    Generates judge scores for given responses based on the judges knowledge.
    Args:
        judge: The model or function used to generate judge scores.
        tokenizer: The tokenizer used to process the input and output text.
        system_prompt (str): The system prompt to be used in the conversation.
        questions (list of str): A list of questions to be evaluated.
        responses (list of str): A list of responses to the questions.
    Returns:
        list of str: A list of judge scores for each response.
    """
    messages = []
    for question, response in zip(questions, responses):
        prompt = get_prompt_wo_reference(question, response)
        conv = [
            {"role": "user", "content": prompt}
        ]
        if system_prompt != "":
            conv.insert(0, {"role": "system", "content": system_prompt})
        messages.append(conv)
    
    chat_convs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate outputs
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    outputs = judge.generate(chat_convs, sampling_params)
    judge_scores = [output.outputs[0].text for output in outputs]

    return judge_scores


def get_judge_scores_w_options(judge, tokenizer, system_prompt, questions, responses, options, answer_indices):
    """
    Generates judge scores for given responses based on the correct and incorrect options.
    Args:
        judge: The model or function used to generate judge scores.
        tokenizer: The tokenizer used to process the input and output text.
        system_prompt (str): The system prompt to be used in the conversation.
        questions (list of str): A list of questions to be evaluated.
        responses (list of str): A list of responses to the questions.
        options (list of list of str): A list of lists containing options for each question.
        answer_indices (list of int): A list of indices indicating the correct option for each question.
    Returns:
        list of str: A list of judge scores for each response.
    """
    messages = []
    for question, response, option_set, answer_index in zip(questions, responses, options, answer_indices):
        prompt = get_prompt_w_options(question, response, option_set, answer_index)
        conv = [
            {"role": "user", "content": prompt}
        ]
        if system_prompt != "":
            conv.insert(0, {"role": "system", "content": system_prompt})
        messages.append(conv)
    
    chat_convs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate outputs
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    outputs = judge.generate(chat_convs, sampling_params)
    judge_scores = [output.outputs[0].text for output in outputs]

    return judge_scores


def get_judged_accuracy(data, dataset_name, model_name):
    """
    Calculate and print the accuracy of a model based on judge scores.
    Parameters:
        data (list or dict): A list of judge scores or a dictionary containing judge scores.
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model.
    Returns:
        float: The accuracy of the model.

    The function prints the dataset name, model name, number of correct responses, 
    number of incorrect responses, and the calculated accuracy. If a sample has 
    an invalid response, it prints a warning message.
    """

    print(f"\nDataset: {dataset_name}, Model: {model_name}")
    if isinstance(data, list):
        judge_scores = data
    else:
        judge_scores = data[dataset_name]["correct"][model_name]
    
    correct = 0
    incorrect = 0
    for score in judge_scores:
        if score == "1":
            correct += 1
        elif score == "0":
            incorrect += 1
        else:
            print(f"Sample has an invalid response: {score}")

    accuracy = correct / (correct + incorrect) 
    print(f" Number of correct responses: {correct}")
    print(f" Number of incorrect responses: {incorrect}")
    print(f" Accuracy: {accuracy:4.3f}\n")

    return accuracy


def run_judge(use_options, judge_name, data_dir, resp_type, n_gpus):
    """
    Evaluates the correctness of model responses using a specified judge model and updates the results.
    Args:
        use_options (bool): Whether to use multiple-choice options for evaluation.
        judge_name (str): The name of the judge model to be used for evaluation.
        data_dir (str): The directory containing model subdirectories with samples.json files.
        resp_type (str): The type of response to be evaluated (e.g., "response", "generated_response").
        n_gpus (int): The number of GPUs to be used for tensor parallelism.
    Returns:
        None
    This function performs the following steps:
    1. Initializes the tokenizer and judge model.
    2. Iterates over all model subdirectories in the input directory.
    3. Loads the samples.json file for each model.
    4. Checks if the judge was already used for the model and skips if so.
    5. Extracts questions and responses from the samples.
    6. Evaluates the correctness of the responses using the judge model.
    7. Appends the judge scores to the samples.json file.
    8. Computes correctness statistics and updates the correctness_stats.csv file.
    """

    # Initialize the tokenizer and judge
    tokenizer = AutoTokenizer.from_pretrained(judge_name)
    judge = LLM(model=judge_name, max_model_len=4096, tensor_parallel_size=n_gpus, enable_prefix_caching=True)
    if "gemma" not in judge_name:
        if use_options:
            system_prompt = "You are a fair judge, tasked with evaluating the correctness of responses relative to given ground-truth answers."
        else:
            system_prompt = "You are a fair judge, tasked with evaluating the correctness of responses to the best of your abilities."
    else: 
        system_prompt = ""
    short_judge_name = judge_name.split("/")[1]

    # Path for correctness stats CSV
    correctness_stats_path = os.path.join(data_dir, "judge_scores.csv")

    # Iterate over all model subdirectories in the input directory
    for model_dir in os.listdir(data_dir):
        model_path = os.path.join(data_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        print(f"Processing model: {model_dir}")

        # Load samples.json
        samples_file = os.path.join(model_path, "samples.json")
        if not os.path.exists(samples_file):
            continue

        with open(samples_file, 'r') as f:
            samples = json.load(f)

        # Check if the judge was already used for this model
        sample = samples[0]
        if f"score_{short_judge_name}" in sample:
            print(f"Model {model_dir} already processed. Skipping...")
            continue

        # Extract questions and filtered responses
        questions = [sample["question"] for sample in samples]
        responses = [sample[resp_type] for sample in samples]
        if use_options:
            options = [sample["options"] for sample in samples]
            answer_indices = [sample["answer_index"] for sample in samples]

        # Get judge scores
        if use_options:
            scores = get_judge_scores_w_options(judge, tokenizer, system_prompt, questions, responses, options, answer_indices)
        else:
            scores = get_judge_scores_wo_references(judge, tokenizer, system_prompt, questions, responses)

        # Append scores to each sample in samples.json
        for sample, score in zip(samples, scores):
            sample[f"score_{short_judge_name}"] = score

        # Save updated samples.json
        with open(samples_file, 'w') as f:
            json.dump(samples, f, indent=2)

        # Compute correctness statistics
        accuracy = get_judged_accuracy(scores, "MMLU-Pro-Freeform", model_dir)

        # Load or create the correctness stats table
        correctness_stats_path = os.path.join(data_dir, "correctness_stats.csv")
        if os.path.exists(correctness_stats_path):
            stats_df = pd.read_csv(correctness_stats_path)
        else:
            # Initialize an empty DataFrame with at least the "model" column
            stats_df = pd.DataFrame(columns=["model"])

        # Ensure the "model" column is set as the index for easier lookups
        if "model" not in stats_df.columns:
            stats_df["model"] = None
        stats_df.set_index("model", inplace=True)

        # Ensure the judge column exists
        judge_column = f"accuracy_{short_judge_name}"
        if judge_column not in stats_df.columns:
            stats_df[judge_column] = None  # Add the column with default None values

        # Update the accuracy for the current judge
        if model_dir not in stats_df.index:
            # Add a new row with None for all columns
            new_row = {col: None for col in stats_df.columns}
            stats_df.loc[model_dir] = new_row

        stats_df.at[model_dir, judge_column] = accuracy

        # Reset the index to ensure "model" remains the first column in the CSV
        stats_df.reset_index(inplace=True)

        # Save the updated table
        stats_df.to_csv(correctness_stats_path, index=False)

        print(f"Processed model: {model_dir}, Accuracy: {accuracy:.2f}")
            

if __name__ == '__main__':
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Compute or load difference scores for model responses.")
     
    parser.add_argument(
        "--judge_model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="The name of the model used for judging the difference between responses."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path for the data."
    )
    parser.add_argument(
        "--resp_type",
        type=str,
        default="filtered_resps",
        help="The type of response to evaluate."
    )
    parser.add_argument(
        "--use_options",
        action="store_true",
        help="Use options for judging correctness."
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="The amount of GPUs available."
    ) 

    args = parser.parse_args()
    judge_name = args.judge_model
    data_dir = args.data_path
    use_options = args.use_options
    resp_type = args.resp_type
    n_gpus = args.n_gpus
    
    run_judge(use_options, judge_name, data_dir, resp_type, n_gpus)
