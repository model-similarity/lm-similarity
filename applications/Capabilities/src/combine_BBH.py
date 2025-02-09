import argparse
import os
import json
from pathlib import Path


argparser = argparse.ArgumentParser(description="Process different ML benchmark datasets")
argparser.add_argument('--data', type=str,
                        required=True, help='Type of dataset to process')
args = argparser.parse_args()


# Define folder paths
input_folders = os.listdir("cleaned_dumps")  # Replace with your folder paths
input_folders = ["cleaned_dumps/" + folder for folder in input_folders if args.data in folder and folder!='OPENLLMLB_MCQ_BBH']
output_folder = f"cleaned_dumps/OPENLLMLB_MCQ_{args.data}"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all unique file names across the input folders
all_filenames = set(
    filename for folder in input_folders for filename in os.listdir(folder) if filename.endswith('.json')
)

# Combine JSON files with the same name

global_q_count = 0
q_tracker = {}
for filename in all_filenames:
    combined_data = []
    
    for folder in input_folders:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):  # Check if file exists in the folder
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    for question in data:

                        if (question['question'], question['answer']) not in q_tracker:
                            q_tracker[(question['question'], question['answer'])] = global_q_count
                            global_q_count+=1

                        question['question_id'] = q_tracker[(question['question'], question['answer'])]

                    combined_data.extend(data)  # Append lists
                    
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}")
    
    # Write combined data to the output folder
    output_file_path = os.path.join(output_folder, filename)
    print(f"Total {len(combined_data)} questions")
    with open(output_file_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

print(f"Combined JSON files saved in '{output_folder}'\n{global_q_count}")