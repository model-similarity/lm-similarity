#!/bin/bash

# Define lists
datasets=("anli-r2" "sciq" "cola" "ethics-utilitarianism" "sst2" "twitter-sentiment" "boolq" "dream" "mc_taco" "multirc" "quail" "quartz" "social_i_qa" "wic" "cosmos_qa")
weak_models=("meta-llama/Llama-3.2-3B" "google/gemma-2-2b" "Qwen/Qwen2.5-1.5B" "microsoft/phi-2")
strong_models=("meta-llama/Llama-3.1-8B" "google/gemma-2-9b" "Qwen/Qwen2.5-7B")

# Output file
output_file="combinations_d2.txt"

# Remove existing file if it exists
rm -f $output_file

# Function to extract substring after the first slash
sanitize_name() {
    local name="$1"
    if [[ "$name" == */* ]]; then
        echo "${name#*/}"
    else
        echo "$name"
    fi
}

# Generate all combinations with sanitized model names
for dataset in "${datasets[@]}"; do
    for weak_model in "${weak_models[@]}"; do
        sanitized_weak_model=$(sanitize_name "$weak_model")
        for strong_model in "${strong_models[@]}"; do
            sanitized_strong_model=$(sanitize_name "$strong_model")
            # Write to combinations.txt: dataset weak_model strong_model sanitized_weak_model sanitized_strong_model
            echo "$dataset $weak_model $strong_model $sanitized_weak_model $sanitized_strong_model" >> "$output_file"
        done
    done
done

echo "Generated $output_file with all combinations."