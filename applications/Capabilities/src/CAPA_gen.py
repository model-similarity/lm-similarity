import logging
import math
import os
import argparse
import sys
import traceback
import json
import numpy as np
from scipy.spatial.distance import jensenshannon as jsd
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import functools
from lmsim.metrics import CAPA

@dataclass
class MetricsResult:
    score: float

class ModelMetrics:
    def __init__(self, path: str):
        self.name = path.split('/')[-1]
        self.data = self._load_data(path)
        self.subject_metrics = self._calculate_subject_metrics()
        
    @staticmethod
    def _load_data(path: str) -> List[Dict]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            print(path)
            
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum(axis=0)
    
    def _calculate_subject_metrics(self) -> Dict[str, Dict]:
        metrics = {}
        total_correct = 0
        total_questions = 0
        
        for item in self.data:
            subject = item['subject']
            if subject not in metrics:
                metrics[subject] = {
                    'questions': [],
                    'correct': 0,
                    'total': 0
                }
            
            metrics[subject]['questions'].append(item)
            if item['acc'] == 1:
                metrics[subject]['correct'] += 1
                total_correct += 1
            metrics[subject]['total'] += 1
            total_questions += 1
            
        # Calculate accuracies
        for subject in metrics:
            metrics[subject]['accuracy'] = metrics[subject]['correct'] / metrics[subject]['total']
        
        metrics['overall'] = {
            'accuracy': total_correct / total_questions,
            'total': total_questions
        }
        
        return metrics

class MetricsCalculator:
    def __init__(self, model1: ModelMetrics, model2: ModelMetrics, type_m: str):
        self.m1 = model1
        self.m2 = model2
        
        if type_m == "prob":
            self.metric = CAPA()
        else:
            self.metric = CAPA(prob=False)
        
        self.type_m = type_m
        self._validate_models()
    
    def _validate_models(self):
        """Validate that both models have the same subjects and questions"""
        assert self.m1.subject_metrics['overall']['total'] == self.m2.subject_metrics['overall']['total']
        for subject in self.m1.subject_metrics:
            if subject == 'overall':
                continue
            assert subject in self.m2.subject_metrics
            
    def calculate_cobs(self, questions1: List[Dict], questions2: List[Dict]) -> Tuple[float, int, float, float, List[int]]:

        questions1.sort(key=lambda x: x['question_id'])
        questions2.sort(key=lambda x: x['question_id'])

        output_a = []
        output_b = []
        gt = []

        
        for q1, q2 in zip(questions1, questions2):
            assert q1['question_id'] == q2['question_id']

            if type(q1['answer']) == int:
                correct_idx1 = q1['answer']
            else:
                correct_idx1 = q1['choices'].index(q1['answer'])
            
            gt = gt.append(correct_idx1)

            if self.type_m == "prob":

                q1_softmax = self.m1.softmax(q1['logits'])
                q2_softmax = self.m2.softmax(q2['logits'])

                output_a.append(q1_softmax)
                output_b.append(q2_softmax)

            else:
                len_choices = len(q1['choices'])
                op_a = np.zeros(len_choices)
                op_b = np.zeros(len_choices)


                if q1['logits']:
                    choice_a = np.argmax(q1["logits"]) 
                    choice_b = np.argmax(q2["logits"])

                    op_a[choice_a] = 1
                    op_b[choice_b] = 1

                else:
                    choice_a = q1['choices'].index(q1['pred'])
                    choice_b = q2['choices'].index(q2['pred'])

                    
                    op_a[choice_a] = 1
                    op_b[choice_b] = 1
                
                output_a.append(op_a)
                output_b.append(op_b)
            
        similarity = self.metric.compute_k(output_a, output_b, gt)
        
        # print(same, total, m1_corr, m2_corr, len(op_len))
        return MetricsResult(score=similarity)

    
def calculate_and_save_metrics(calculator: MetricsCalculator, 
                             model1: ModelMetrics, 
                             model2: ModelMetrics, 
                             path1: str, 
                             path2: str, 
                             sortby: str, 
                             diffpath: str) -> Dict:
    """Calculate metrics for model pairs and save results to files"""
    
    # Create output directory
    os.makedirs(f"{diffpath}/{sortby}/", exist_ok=True)
    
    # Get model names from paths
    model1_name = path1.split('/')[-1]
    model2_name = path2.split('/')[-1]
    
    # Calculate metrics for each subject
    subject_results = {}
    overall_questions = []
    
    for subject in model1.subject_metrics:
        if subject == 'overall':
            continue
            
        m1_questions = model1.subject_metrics[subject]['questions']
        m2_questions = model2.subject_metrics[subject]['questions']
        
        # Calculate metrics
        kappa_score = calculator.calculate_cobs(
            m1_questions, m2_questions)
        

        # Store results
        subject_results[subject] = {
            "acc1": model1.subject_metrics[subject]['accuracy'],
            "acc2": model2.subject_metrics[subject]['accuracy'],
            "kappa": {"score": kappa_score.score}
        }
        
        overall_questions.extend(m1_questions)
    
    # Calculate overall metrics
    overall_score = calculator.calculate_cobs(
        overall_questions, 
        [q for s in model2.subject_metrics if s != 'overall' for q in model2.subject_metrics[s]['questions']])

    # Add overall results
    subject_results["overall"] = {
        "acc1": model1.subject_metrics['overall']['accuracy'],
        "acc2": model2.subject_metrics['overall']['accuracy'],
        "kappa": {"score": overall_score.score}
    }
    
    # Save results for both model orderings
    for order in [(model1_name, model2_name, False), (model2_name, model1_name, True)]:
        model_a, model_b, swap = order
        output_path = f"{diffpath}/{sortby}/subjectwise___{model_a}___{model_b}.txt"
        
        result_dict = {}
        for subject, metrics in subject_results.items():
            result_dict[subject] = {
                "acc1": metrics["acc2" if swap else "acc1"],
                "acc2": metrics["acc1" if swap else "acc2"],
                "kappa": metrics["kappa"]
            }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=4)
    
    return subject_results


def process_model_pair(args: Tuple[str, str, str, str, str]) -> None:
    path1, path2, sortby, diffpath, type_m = args
    
    if os.path.isdir(path1) or os.path.isdir(path2):
        return
        
    model1 = ModelMetrics(path1)
    model2 = ModelMetrics(path2)
    calculator = MetricsCalculator(model1, model2, type_m)
        
        # Calculate metrics and save results
    results = calculate_and_save_metrics(calculator, model1, model2, path1, path2, 
                                          sortby, diffpath)
        
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=str)
    parser.add_argument('--type', type=str, default='prob', 
                       help='type of metric prob or not')
    args = parser.parse_args()
    
    dirs = args.dirs
    model_families_skip = ['.DS_Store']

    try:
        dirs_list = os.listdir(dirs)

        sub_dirs_list = []
        for models in dirs_list:
            if [skip for skip in model_families_skip if skip in models]:
                continue
            else:
                sub_dirs_list.append(models)

        process_args = []

        print(len(sub_dirs_list), len(dirs_list))

        for i in range(len(sub_dirs_list)):
            for j in range(i, len(sub_dirs_list)):
                path1 = f"{dirs}/{sub_dirs_list[i]}"
                path2 = f"{dirs}/{sub_dirs_list[j]}"
                diffpath = f'{dirs}/diffs'
                sortby = f'kappa_{args.type}'
                
                process_args.append((path1, path2, sortby, diffpath, args.type))
        
        print(len(process_args))
        # Create the diffs directory if it doesn't exist
        os.makedirs(f'{dirs}/diffs', exist_ok=True)
        
        # Process model pairs in parallel
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_model_pair, process_args),
                total=len(process_args),
                desc="Processing model pairs"
            ))

    except Exception as e:

        print(e)

if __name__ == "__main__":
    main()