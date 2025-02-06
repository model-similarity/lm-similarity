# Application: LM-as-Annotators

This directory contains the steps to reproduce the experiments of Chapter 4 "Learning from Complementary Knowledge of LM Annotators" in [paper](). 

We use the open-source reproduction of weak-to-strong generalization experiments by [this blog post](https://blog.eleuther.ai/weak-to-strong/), based in part on [openai/weak-to-strong](https://github.com/openai/weak-to-strong).


## Running experiments
The following explanation assumes that the python code is executed from the `src/` directory. 

Basic invocation: 

`python compute_w2s.py --dataset sciq --run_name my_run`

For reproducing paper experiments, run `w2s/gen_combi.sh`  to obtain full list of model pairs and datasets. Example run used in paper is provided in `run.sh`.


## Existing results and data

Results from our experiments, including plots and sample-wise predictions will be stored in `src/results/epochs_3/`. Plotting code is available in `src/visualizations.py`, `src/scatters.ipynb`. Unfortunately the ICML supplementary size limit was 100MB so we could not share sample-wise predictions for these experiments (6GB+). We will release them upon publication.

## Troubleshooting

The models used are often fated, see [here](https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user) for details.