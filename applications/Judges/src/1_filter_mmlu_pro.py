import json
import os
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import login
import datasets

from utils import load_settings


# Define the prompts for the model
coarse_filter_prompt = """Your task is to review a series of multiple-choice questions and evaluate their ability to
be answered without the provided answer choices. For questions that begin with an incomplete
sentence (e.g., "During swallowing, ..."), use your knowledge to attempt to complete the
sentence accurately. For direct questions that ask for specific information or identification
(e.g., "Which of the following structures is part of the small intestine?"), assess whether the
question is formulated clearly enough that an informed answer can be given without seeing the
multiple-choice options. For mathematical or analytical questions (e.g., "Find all cosets of
the subgroup 4Z of 2Z"), determine if the question provides enough context and information for a
solution to be formulated without additional options.

Please follow this format for your evaluation:

QUESTION: [Insert the question here] 

VERDICT: Respond with "YES" if the question is clear and can be directly answered based on its 
content alone, or "NO" if it relies on the answer choices to be understood or answered. Your 
response should include only the verdict without any justification or reasoning."""

fine_filter_prompt = """You will assign a numerical score from 1 to 10 based on how confidently it can be answered
without the choices. The scoring criteria are as follows:

1: The question is entirely dependent on its choices for an answer, making it impossible to
answer without them. Example: ‘Which of the following statements is correct?’

10: The question can be easily and confidently answered based solely on the question stem,
without any need to refer to the provided options. Example: ‘What is the first law of
thermodynamics in physics?’ 

Intermediate Scores:

2-4: The question stem gives very little information and is highly reliant on the choices for
context. Example: ‘Which of these is a prime number?’ 'The ________ perspective on sustainability 
resulted from growth models that analysed the carrying capacity of the planet, overall concluding 
that the finite capacity of the earth and_______, ________ and _______ by current and past 
generations could reduce quality of life for future generations.'

5: The question provides some context or information, that gives a moderate possibility to 
answer the question. Example: ‘Which of the following best describes the structure that collects 
urine in the body?’ Example: 

6: The question provides a good amount of context or information, that gives a moderate
possibility to answer the question. Example: ‘Statement 1 | A factor group of a non-Abelian
group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of
G, then K is a normal subgroup of G.’

7: The question provides a good amount of context or information, that gives a high possibility
to answer the question. Example: ‘The element (4, 2) of Z_12 x Z_8 has order’

8-9: The question provides a good amount of context or information, that gives a high
possibility to answer the question. Example: ‘A "dished face" profile is often associated with’

ONLY GIVE THE VALUE BETWEEN 1-10 AS YOUR ANSWER. DO NOT INCLUDE ANY OTHER INFORMATION IN YOUR
RESPONSE
"""


def get_filter_decisions(llm, dataset, tokenizer, sampling_params, system_prompt=""):
    """
    Generates filter decisions for a given dataset using a language model (LLM).
    Args:
        llm: The language model to use for generating outputs.
        dataset (list): A list of dictionaries, where each dictionary contains a "question" key.
        tokenizer: The tokenizer to use for preparing the input prompts.
        sampling_params: Parameters for sampling during LLM generation.
        system_prompt (str, optional): An optional system prompt to prepend to each message. Defaults to "".
    Returns:
        list: A list of dictionaries, where each dictionary contains the original dataset fields
              along with "llm_judge_coarse" and "llm_judge_fine" keys representing the LLM's decisions.
    """

    results = []
    questions = [sample["question"] for sample in dataset]

    # Generate the prompts
    coarse_prompts = [
        coarse_filter_prompt + f"\n\nQuestion: {q} \n\nVerdict:" for q in questions
    ]
    messages_coarse = []
    for prompt in coarse_prompts:
        messages_coarse.append([{"role": "user", "content": prompt}])
        if system_prompt:
            messages_coarse[-1].insert(0, {"role": "system", "content": system_prompt})

    fine_prompts = [
        fine_filter_prompt + f"\n\nQuestion: {q} \n\nScore:" for q in questions
    ]
    messages_fine = []
    for prompt in fine_prompts:
        messages_fine.append([{"role": "user", "content": prompt}])
        if system_prompt:
            messages_fine[-1].insert(0, {"role": "system", "content": system_prompt})

    # Tokenize the conversations
    coarse_convs = tokenizer.apply_chat_template(
        messages_coarse, tokenize=False, add_generation_prompt=True
    )
    fine_convs = tokenizer.apply_chat_template(
        messages_fine, tokenize=False, add_generation_prompt=True
    )

    # Generate the outputs
    coarse_outputs = llm.generate(coarse_convs, sampling_params)
    fine_outputs = llm.generate(fine_convs, sampling_params)

    # Extract the results
    coarse_results = [output.outputs[0].text for output in coarse_outputs]
    fine_results = [output.outputs[0].text for output in fine_outputs]

    for sample, coarse_output, fine_output in zip(
        dataset, coarse_results, fine_results
    ):
        coarse_decision = coarse_output.strip()
        if coarse_decision not in ["YES", "NO"]:
            coarse_decision = "NO"

        try:
            fine_decision = int(fine_output.strip())
        except ValueError:
            fine_decision = 1  # Default to lowest score
        else:
            fine_decision = max(1, min(10, fine_decision))  # Clamp to 1-10

        sample["llm_judge_coarse"] = coarse_decision
        sample["llm_judge_fine"] = fine_decision
        results.append(sample)

    return results


def main(args):
    data_path = args.data_path
    model_name = args.model
    n_gpus = args.n_gpus
    threshold = args.threshold

    output_file = os.path.join(data_path, "mmlu-pro_test_osq_filter.json")

    # Load the dataset
    data = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test").to_list()

    # Login to the hugging face hub
    settings = load_settings("settings.yaml")
    hf_token = settings["huggingface"]["api_token"]
    login(token=hf_token)

    if "Qwen" in model_name:
        system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )
    else:
        system_prompt = ""

    # Load the model and tokenizer
    llm = LLM(model=model_name, tensor_parallel_size=n_gpus, enable_prefix_caching=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(max_tokens=3, temperature=0.0)

    # Get the filter decisions
    data = get_filter_decisions(llm, data, tokenizer, sampling_params, system_prompt)

    # Save the filtered dataset
    os.makedirs(data_path, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    # Filter the dataset
    filtered_questions = [
        sample
        for sample in data
        if sample["llm_judge_coarse"] == "YES" or sample["llm_judge_fine"] >= threshold
    ]
    print(f"Filtered questions: {len(filtered_questions)} / {len(data)}")

    # Save filtered question_ids
    filtered_question_ids = [sample["question_id"] for sample in filtered_questions]
    with open(os.path.join(data_path, "filtered_question_ids.txt"), "w") as f:
        for question_id in filtered_question_ids:
            f.write(f"{question_id}\n")


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the questions of MMLU-Pro and filter the ones that can be answered in free-form."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="The name of the model used to perform the filtering.",
    )
    parser.add_argument("--data_path", type=str, help="The path for the data.")
    parser.add_argument(
        "--n_gpus", type=int, default=1, help="The amount of GPUs available."
    )
    parser.add_argument(
        "--threshold", type=int, default=5, help="The threshold for the fine filtering."
    )

    args = parser.parse_args()

    main(args)
