import argparse
import json
import os
from functools import reduce
from tqdm import tqdm

def process_json_files(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    q_track = []

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.json'):
            with open(os.path.join(input_folder, filename), 'r') as f:
                data = json.load(f)
            temp = set()
            for item in data:
                temp.add((item['question'], item['answer'], item['subject'], len(item['logits'])))
            q_track.append(temp)
    common_elements = reduce(lambda x, y: x.intersection(y), q_track)
    print("Common Qusetion & Answers: ", len(common_elements))

    # Process each file
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.json'):
            with open(os.path.join(input_folder, filename), 'r') as f:
                data = json.load(f)
            common_dicts = []
            added_pairs = []

            for item in data:
                if (item['question'], item['answer'], item['subject'], len(item['logits'])) in common_elements:
                    if (item['question'], item['answer'], item['subject'], len(item['logits'])) not in added_pairs:
                        common_dicts.append(item)
                        added_pairs.append((item['question'], item['answer'], item['subject'], len(item['logits'])))

            common_dicts = sorted(common_dicts, key=lambda x: (x['question'], x['subject'], len(x['logits']), x['answer']))

            for idx, item in enumerate(common_dicts):
                item['question_id'] = idx
            
            # print(len(common_dicts))
            
            # Write to output file
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as f:
                json.dump(common_dicts, f, indent=2)

def verify_files(input_folder):
    count= 0 
    filename = os.listdir(input_folder)[0]
    with open(os.path.join(input_folder, filename), 'r') as f:
        data = json.load(f)
    
    q_no_id = {}
    for idx, item in enumerate(data):
        if  item['question_id'] in q_no_id.values():
            print("sus")
        q_no_id[idx] =  item['question_id']


    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.json'):
            with open(os.path.join(input_folder, filename), 'r') as f:
                data = json.load(f)
                
                for idx, item in enumerate(data):
                    if q_no_id[idx] != item['question_id']:
                        print(q_no_id[idx], item['question'],  item['answer'])
                        print(filename)
                        count+=1
                        break
                     
    
    print(count)


    q_track = []

    for filename in (os.listdir(input_folder)):
        if filename.endswith('.json'):
            with open(os.path.join(input_folder, filename), 'r') as f:
                data = json.load(f)
            
            temp = set()
            for item in data:
                temp.add((item['question'], item['answer'], item['subject'], len(item['logits'])))
            print(len(data), len(temp))
            q_track.append(temp)
        
    common_elements = reduce(lambda x, y: x.intersection(y), q_track)
    print(len(common_elements))



parser = argparse.ArgumentParser()
parser.add_argument('dirs', type=str)
args = parser.parse_args()

input_folder = args.dirs
output_folder = args.dirs + '_cleaned'
process_json_files(input_folder, output_folder)
print("verifying files")
verify_files(output_folder)