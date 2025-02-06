source ~/miniforge3/etc/profile.d/conda.sh # equivalent to conda init
conda activate diff
export HOME=""
export SOFT_FILELOCK=1
# optionally parse args
dataset="$1"
run_name="$2"
weak_model_name="$3"
strong_model_name="$4"

# execute python script
python compute_w2s.py --dataset="$dataset" --run_name="$run_name" --weak_model_name="$weak_model_name" --strong_model_name="$strong_model_name"