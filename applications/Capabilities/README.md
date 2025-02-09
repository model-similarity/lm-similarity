# Application: LM Capabilities

This directory contains the steps to reproduce the experiments of Chapter 5 "Models are making more similar mistakes as capabilities increase" in [paper](https://arxiv.org/abs/2502.04313). 



## Running experiments
The following explanation assumes that the python code is executed from outside the `src/` directory. 

Basic invocation: 

`python src/CAPA_gen.py dataset_path --type metric_type`

For reproducing paper experiments, run `run_CAPA_compute.sh` to obtain full list of model pairs and datasets.


## Existing results and data

Results from our experiments, including plots and sample-wise predictions will be stored in `dataset_path/diffs`. 

## Troubleshooting

The models used are often fated, see [here](https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user) for details.