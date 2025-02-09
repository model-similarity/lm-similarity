import os
import shutil
# Define the base directory

bbh_data = [
    "samples_leaderboard_bbh_boolean_expressions",
    "samples_leaderboard_bbh_formal_fallacies",
    "samples_leaderboard_bbh_causal_judgement",
    "samples_leaderboard_bbh_date_understanding",
    "samples_leaderboard_bbh_disambiguation_qa",
    "samples_leaderboard_bbh_geometric_shapes",
    "samples_leaderboard_bbh_hyperbaton",
    "samples_leaderboard_bbh_logical_deduction_five_objects",
    "samples_leaderboard_bbh_logical_deduction_seven_objects",
    "samples_leaderboard_bbh_logical_deduction_three_objects",
    "samples_leaderboard_bbh_movie_recommendation",
    "samples_leaderboard_bbh_navigate",
    "samples_leaderboard_bbh_penguins_in_a_table",
    "samples_leaderboard_bbh_reasoning_about_colored_objects",
    "samples_leaderboard_bbh_ruin_names",
    "samples_leaderboard_bbh_salient_translation_error_detection",
    "samples_leaderboard_bbh_snarks",
    "samples_leaderboard_bbh_sports_understanding",
    "samples_leaderboard_bbh_temporal_sequences",
    "samples_leaderboard_bbh_tracking_shuffled_objects_five_objects",
    "samples_leaderboard_bbh_tracking_shuffled_objects_seven_objects",
    "samples_leaderboard_bbh_tracking_shuffled_objects_three_objects",
    "samples_leaderboard_bbh_web_of_lies",
]
model_families_skip = ['.DS_Store']

for bbh in bbh_data:
    dir_name_bbh = bbh.split('samples_leaderboard_bbh_')[1]
    base_dir = f'cleaned_dumps/OPENLLMLB_MCQ_BBH_{dir_name_bbh}'
    os.makedirs(base_dir, exist_ok=True)
    files_dir = 'dumps/openllmlb'
    for subfolder in os.listdir(files_dir):
        subfolder_path = os.path.join(files_dir, subfolder)
        # Ensure the current item is a directory (e.g., A1, A2, etc.)
        if len([skip for skip in model_families_skip if skip in subfolder])>0:
            print("skipping: ", subfolder)
            continue
        if os.path.isdir(subfolder_path):
            clean_folder = os.path.join(subfolder_path, 'clean')
            
            # Check if the 'clean' folder exists
            if os.path.isdir(clean_folder):
                # List all files in the clean folder
                json_files = [f for f in os.listdir(clean_folder) if (f.endswith('.json') or f.endswith('.jsonl')) and bbh in f]

                
                # Check if there is exactly one JSON file
                if len(json_files) == 1:
                    json_file_path = os.path.join(clean_folder, json_files[0])
                    # Define the new filename based on the subfolder name (e.g., A1_json.json)
                    new_filename = f"{subfolder}.json"
                    new_file_path = os.path.join(base_dir, new_filename)
                    
                    # Copy the JSON file and rename it
                    shutil.copy(json_file_path, new_file_path)
                    # print(f"Copied and renamed: {json_file_path} -> {new_file_path}")
                else:
                    print(json_files)
                    print(f"{subfolder_path} -  {len(json_files)}.")
            else:
                print(f"BBH Clean folder not found in: {subfolder_path}")


base_dir = f'cleaned_dumps/OPENLLMLB_MCQ_MMLU'
os.makedirs(base_dir, exist_ok=True)
files_dir = 'dumps/openllmlb'
for subfolder in os.listdir(files_dir):
    subfolder_path = os.path.join(files_dir, subfolder)
    # Ensure the current item is a directory (e.g., A1, A2, etc.)
    if len([skip for skip in model_families_skip if skip in subfolder])>0:
        print("skipping: ", subfolder)
        continue
    if os.path.isdir(subfolder_path):
        clean_folder = os.path.join(subfolder_path, 'clean')
        
        # Check if the 'clean' folder exists
        if os.path.isdir(clean_folder):
            # List all files in the clean folder
            json_files = [f for f in os.listdir(clean_folder) if (f.endswith('.json') or f.endswith('.jsonl')) and 'mmlu' in f]

            
            # Check if there is exactly one JSON file
            if len(json_files) == 1:
                json_file_path = os.path.join(clean_folder, json_files[0])
                # Define the new filename based on the subfolder name (e.g., A1_json.json)
                new_filename = f"{subfolder}.json"
                new_file_path = os.path.join(base_dir, new_filename)
                
                # Copy the JSON file and rename it
                shutil.copy(json_file_path, new_file_path)
            else:
                print(json_files)
                print(f"{subfolder_path} -  {len(json_files)}.")
        else:
            print(f"MMLU Clean folder not found in: {subfolder_path}")