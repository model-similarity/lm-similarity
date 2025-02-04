# Examples

This is a folder showcases how **lm-sim** can be used. 

### Requirements 
For the examples to run you must have installed:
- datasets(>=3.2.0) 

### Access
For some models you need to request access on the hf, and specify you API token **locally** in `settings.yaml`. 

### Usage
Open the corresponding benchmark folder and execute:
```
python compute_sim.py
```

If you want to test different models than the default, pass them as arguments:

```
python compute_sim.py --model_a "Qwen/Qwen2.5-32B-Instruct" --model_b "meta-llama/Meta-Llama-3.1-8B-Instruct"
```


