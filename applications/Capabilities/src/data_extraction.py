import os
import json
import argparse
from typing import Callable, Dict, List, Any

from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, path: str, dataset_type: str, startswith: str = None):
        """
        Initialize dataset processor with common configuration
        
        Args:
            path (str): Root path to dataset files
            dataset_type (str): Type of dataset (bbh, mmlu_pro)
            startswith (str, optional): Prefix for files to process
        """
        self.path = path
        self.dataset_type = dataset_type
        self.startswith = startswith or f"samples_{dataset_type}"
        self.extractors = {
            'bbh': self.extract_fields_bbh,
            'mmlu_pro': self.extract_fields_mmlu_pro
        }

    def _load_json_file(self, filepath: str) -> List[Dict]:
        """
        Robustly load JSON data from file
        
        Args:
            filepath (str): Path to JSON file
        
        Returns:
            List of loaded data entries
        """
        with open(filepath, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f]

    def extract_fields_bbh(self, data: Dict, question_map: Dict[str, int]) -> Dict:
        """Extract fields for BBH dataset"""
        question = data['doc']['input']
        try:
            question_id = data['doc_id']
        except KeyError:
            print("question_id")
            question_id = 144 
        
        result = {
            'question_id': question_id,
            'question': question,
            'choices': [args['arg_1'].strip() for en, args in data['arguments'].items()],
            'answer': data['doc']['target'],
            'subject': data['doc'].get('subject', 'all')
        }

        log_probs = [float(resp[0][0]) for resp in data['resps']]
        result['logits'] = log_probs
        
        max_log_prob_index = log_probs.index(max(log_probs))
        result['pred'] = result['choices'][max_log_prob_index]
        result['acc'] = int(result['pred'] == result['answer'])
        
        return result

    def extract_fields_mmlu_pro(self, data: Dict, question_map: Dict[str, int]) -> Dict:
        """Extract fields for MMLU Pro dataset"""
        question = data['doc']['question'].strip()
        try:
            question_id = question_map[question]
        except KeyError:
            if 'Poisson process' in question:
                question_id = 11581
            elif '200P + 8,000' in question:
                question_id = 1535

            else:
                print(question)
                question_id = 999999 
        
        
        result = {
            'question_id': question_id,
            'question': question,
            'choices': data['doc']['options'],
            'answer': data['doc']['answer_index'],
            'subject': data['doc']['category'],
            'src': data['doc']['src']
        }

        log_probs = [float(resp[0][0]) for resp in data['resps']]
        result['logits'] = log_probs
        
        max_log_prob_index = log_probs.index(max(log_probs))
        result['acc'] = int(max_log_prob_index == result['answer'])
        
        return result

    def process_directory(self) -> None:
        """
        Process dataset files for a specific directory
        
        Dispatches to appropriate extraction method based on dataset type
        """

        model_1_path = os.path.join(self.path, os.listdir(self.path)[1])
        all_questions = self._extract_questions(model_1_path)
        question_map = {q: i for i, q in enumerate(all_questions)}

        for dir_path in tqdm(os.listdir(self.path)):
            full_dir_path = os.path.join(self.path, dir_path)
            
            # Skip non-directories and hidden directories
            if dir_path.startswith(".") or not os.path.isdir(full_dir_path):
                continue

            self._process_directory(full_dir_path, question_map)

    def _process_directory(self, dir_path: str, question_map: Dict[str, int]) -> None:
        """Process BBH dataset directory"""
        # Extract global questions map first
        
        clean_path = os.path.join(dir_path, "clean")
        os.makedirs(clean_path, exist_ok=True)
        
        try:
            file = sorted([f for f in os.listdir(dir_path) if f.startswith(self.startswith)])[-1]
        except Exception as e:
            print("missing file")
            return 
        filepath = os.path.join(dir_path, file)
        
        data = self._load_json_file(filepath)
        clean_qs = [self.extractors[self.dataset_type](q, question_map) for q in data]
        clean_qs = sorted(clean_qs, key=lambda x: x['question_id'])
        
        with open(os.path.join(clean_path, file), "w") as f:
            json.dump(clean_qs, f, indent=4)

    def _extract_questions(self, dir_path: str) -> List[str]:
        """Extract questions for BBH dataset"""
        try:
            file = sorted([f for f in os.listdir(dir_path) if f.startswith(self.startswith)])[-1]
        except Exception as e:
            print("missing file")
            return []

        data = self._load_json_file(os.path.join(dir_path, file))
        try:
            questions = [entry['doc']['input'].strip() for entry in data]
        except KeyError:
            questions = [entry['doc']['question'].strip() for entry in data]
        return questions

def main():
    argparser = argparse.ArgumentParser(description="Process different ML benchmark datasets")
    argparser.add_argument('path', type=str, help='Root path to dataset')
    argparser.add_argument('--dataset', type=str, choices=['bbh', 'mmlu_pro'], 
                            required=True, help='Type of dataset to process')
    argparser.add_argument('--startswith', type=str, 
                            help='Optional file prefix override')
    args = argparser.parse_args()

    processor = DatasetProcessor(
        path=args.path, 
        dataset_type=args.dataset, 
        startswith=args.startswith
    )
    processor.process_directory()

if __name__ == "__main__":
    main()