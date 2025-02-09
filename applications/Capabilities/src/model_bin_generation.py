import argparse
import os
import json
from typing import Dict, List
import math
import numpy as np

class ModelMetrics:
    def __init__(self, path: str):
        self.name = path.split('/')[-1]
        self.data = self._load_data(path)
        self.total = 0
        self.accuracy = self._calculate_subject_metrics()
        
    @staticmethod
    def _load_data(path: str) -> List[Dict]:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _calculate_subject_metrics(self) -> Dict[str, Dict]:
        total_correct = 0
        
        for item in self.data:
            if item['acc'] == 1:
                total_correct += 1
            self.total += 1
        return total_correct / self.total

parser = argparse.ArgumentParser()
parser.add_argument('dirs', type=str)
args = parser.parse_args()
base_dir = args.dirs


model_binning  = []
total_q = set()
dataset = os.listdir(base_dir)
for model in dataset:
    path1 = os.path.join(base_dir, model)
    if os.path.isdir(path1):
        continue
    model_obj = ModelMetrics(path1)
    total_q.add(model_obj.total)
    model_binning.append(model_obj)


print(len(model_binning))
print(total_q)
percentiles = np.linspace(0, 100, 5 + 1)[1:-1]  # exclude 0 and 100
percentile_thresholds = np.percentile([acc.accuracy for acc in model_binning], percentiles)
print(percentile_thresholds)
# Create bins dictionary using percentile ranges
bins = {}
diff_bins = {}
for i in range(5):
    start_percentile = i * (100 // 5)
    end_percentile = (i + 1) * (100 // 5)
    bin_key = f"{start_percentile}-{end_percentile}%"
    bins[bin_key] = []

for model in model_binning:
    # print(model.name.split('/')[0])
    assigned = False
    for i, threshold in enumerate(percentile_thresholds):
        if model.accuracy <= threshold:
            bin_key = f"{i * (100 // 5)}-{(i + 1) * (100 // 5)}%"
            bins[bin_key].append(model.name)
            assigned = True
            break
    if not assigned:  # Handle the last bin
        bin_key = f"{(5 - 1) * (100 // 5)}-{100}%"
        bins[bin_key].append(model.name)

print([len(v) for k,v in bins.items()])
print(sum([len(v) for k,v in bins.items()]))


output_file_path = os.path.join(base_dir, "bins", "fleiss_bins.json")
with open(output_file_path, 'w') as f:
        json.dump(bins, f, indent=4)