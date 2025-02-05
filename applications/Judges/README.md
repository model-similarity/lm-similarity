# Application: LLM-as-a-Judge 

This directory contains the steps to reproduce the experiments of Chapter 3 "Affinity Bias in AI Judges".

The following explanation assumes that the python code is executed from the `src/` directory. 

## Access
Add your personal Hugging Face API key to the `settings.yaml` file in order to load the data from Hugging Face Hub. 

## 1. Filter MMLU-Pro

The indices that correspond to questions that can be answered without access to the reference answers can be obtained by running the script `1_filter_mmlu_pro.py`. By default, it runs a `Qwen2.5-32B-Instruct` language model to filter questions. It writes a list of question IDs to keep in a file called `filtered_question_ids.txt`. If you do not want to filter MMLU-Pro yourself, you can also use the file we provided in `data/` folder.

## 2. Compute Similarities Between Judges and Evaluated Models

To compute the similarity metrics between judges and evaluated models, you have to call:

```
python 2_calculate_similarities.py 
```
By default $k_p$, $k_p$ discrete, and error consistency are computed. If you want to compute only one of them pass via:
```
python 2_calculate_similarities.py --metrics kappa_p
```
Computed simlarities will be automatically stored in `../output/sim/` folder.

For most models the datafiles are automatically downloaded from the Open LLM Leaderboard on Hugging Face. Unfortunately, it is not possible to load the data automatically for some LMs. For the following models, the JSON logs have to be downloaded manually from `https://huggingface.co/open-llm-leaderboard` and saved to `../data/leaderboard_mmlu_pro`:

```
google/gemma-2-2b-it
google/gemma-2-9b
google/gemma-2-9b-it
```

## 3. Perform Free-Form Inference on MMLU-Pro

The free-form generation is done using a custom task for the LM-Eval Harness. For simplicity, we generate responses for all questions in the test sets of MMLU-Pro and do the filtering afterward. To do so, you have to move the `mmlu_pro_free` directory into the `tasks` directory of your LM-Eval Harness installation. Additionally, we use a custom regex filtering method. It is implemented in the class `FreeformRegexFilter` in the file `mmlu_pro_free/extraction.py` and needs to be moved to the corresponding `extraction.py` of LM-Eval Harness.

Once these preparations are done, you can generate the open-style responses by calling the new task in the LM-Eval Harness. A bash script that will need some modifications depending on your hardware setup can be found in `3_run_mmlu_free.sh`. It contains the list of models that we evaluated and shows how to call the harness. It makes sense to split the generations into multiple calls and adjust the number of GPUs depending on the models you are currently running and the hardware resources available.

## 4. Filter and Merge LM-Eval Harness Generations

MMLU-Pro is split into multiple topics, and the LM-Eval Harness creates a new output directory for every one of them after a model is evaluated. Hence, we want to merge these samples from different categories and only keep the information relevant to our analysis. Additionally, we need to apply filtering to only keep open-style questions:

```
python 4_filter_and_merge.py --input_dir $path_to_lm_eval_output --output_dir $output_path --filter_path $path_to_filtered_question_ids.txt
```

## 5. Run LLM-as-a-Judge on Filtered MMLU-Pro

After the free-form responses are generated, we can rate their correctness using an LLM-as-a-judge. To do so, call `5_llm_judge.py`:

```
python 5_llm_judge.py --judge_model $judge --data_path $path_to_merged_responses --resp_type "filtered_resps" --n_gpus $n_gpus
```

By default, we use the final, short responses at the end of the CoT generation as responses to judge. However, you can also provide the judge with the full CoT responses. To do so, change the flag `--resp_type` to `--resp_type "resps"`. Surprisingly, the LLM-as-judges we evaluated became less accurate and even more biased toward false positives than before when given access to full CoT responses.

### 5.1 Judge Ensemble with Reference Answers

If you want to get a more accurate evaluation of whether a model's free-form responses are correct, you can provide the judge with access to the MCQ reference options. This is done by doing the same call as above and adding the `--use_options` flag. Make sure to create a copy of the unjudged samples to not mix up judge scores with and without knowledge about ground truth.

Once this evaluation has been done for an ensemble of judges, their ratings for each question can be aggregated using majority voting to obtain a good estimation of the real OSQ accuracy of a model. To combine the judgments with reference options, run the following script:

```
python 5.1_ensemble_accuracy.py --data_dir $dir_of_judge_scores_w_options
```

## 6. Reproduce Figures

After the judge scores and similarities are computed, you can reproduce the figures and statistical tests using the Jupyter notebook `6_results_judge_similarities.ipynb`.