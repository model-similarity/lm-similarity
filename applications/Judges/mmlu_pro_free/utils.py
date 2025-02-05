from functools import partial
import re


def transform_cot_to_freeform(cot, options):
    """
    Transforms a Chain-of-Thought for MCQ into one for free-form generation.
    Args:
        cot (str): The original Chain-of-Thought.
        options (List[str]): The multiple-choice options (10 options, A-J).
    Returns:
        str: The transformed Chain-of-Thought.
    """
    # Create a mapping of option letters to their corresponding text
    option_mapping = {chr(65 + i): options[i] for i in range(len(options))}
    
    # Function to replace references in the Chain-of-Thought
    def replace_option_references(match):
        option = match.group(1)  # Get the letter (e.g., A, B, ...)
        return f"({option_mapping.get(option, 'Unknown')})"
    
    # Replace all option references in the Chain-of-Thought
    transformed_cot = re.sub(r"\(([A-J])\)", replace_option_references, cot)
    
    return transformed_cot


def transform_cot_to_freeform_old(cot, answer_index, options):
    """
    Transforms a Chain-of-Thought for MCQ into one for free-form generation.
    Args:
        cot (str): The original Chain-of-Thought.
        answer (str): The correct answer to the question (A to J).
        options (List[str]): The multiple-choice options (10 options, A-J).
    Returns:
        str: The transformed Chain-of-Thought.
    """
    # Replace "(X)" at the end of the CoT with the full answer text
    correct_option = options[answer_index].replace(".", "")
    transformed_cot = cot.rsplit("(", 1)[0].strip() + f" ({correct_option}).\n"

    return transformed_cot

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        cot_content = transform_cot_to_freeform(cot_content, example["options"])
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

def doc_to_target(doc):
    return doc["options"][doc["answer_index"]]

doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")
