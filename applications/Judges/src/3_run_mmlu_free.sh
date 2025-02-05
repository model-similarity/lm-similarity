#!/bin/bash
#SBATCH --job-name=lm_eval_mmlu_free
#SBATCH --array=0-40                 # Number of tasks in the array (one for each model)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=8            # CPUs per task
#SBATCH --mem=32G                    # Memory per task
#SBATCH --time=1-00:00:00            # Maximum runtime
#SBATCH --gres=gpu:XX                # Request GPUs as needed for the model you want to run in this batch
#SBATCH --partition=<Partition>      # Partition to run on

# Activate the conda environment
source ~/.bashrc
conda activate llm_judge

# Array of models for freeform generation
models=(
    "microsoft/phi-4"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "HuggingFaceTB/SmolLM2-135M-Instruct"
    "HuggingFaceTB/SmolLM2-360M-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "tiiuae/Falcon3-1B-Instruct"
    tiiuae/Falcon3-7B-Instruct"
    "tiiuae/Falcon3-10B-Instruct"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-32B"
    Qwen/Qwen2.5-72B"
    "HuggingFaceTB/SmolLM2-135M"
    "HuggingFaceTB/SmolLM2-360M"
    "HuggingFaceTB/SmolLM2-1.7B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.1-70B"
    #"meta-llama/Llama-3.3-70B" - does not exist
    #"tiiuae/Falcon3-1B-Base" - unable to load weights
    "tiiuae/Falcon3-7B-Base"
    "tiiuae/Falcon3-10B-Base"
    "google/gemma-2-2b"
    "google/gemma-2-9b"
    "google/gemma-2-27b"
)

# Get the model filename for the current array task
model=${models[$SLURM_ARRAY_TASK_ID]}
n_gpus=XX

# Run LM Eval Harness
lm-eval --model vllm \
    --model_args pretrained=$model,dtype=auto,gpu_memory_utilization=0.85,enable_prefix_caching=True,tensor_parallel_size=$n_gpus \
    --tasks mmlu_pro_free \
    --batch_size auto \
    --output_path data/mmlu_pro_free \
    --write_out \
    --log_samples 

conda deactivate