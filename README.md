# LM-Similarity

**lm-sim** is a Python module for computing similarity between Language Models and is distributed under the MIT license. 

## Installation

### Dependencies

**lm-sim** requries:
- Python (>=3.9)
- Numpy (>= 1.19.5)

### User installation 
If you already have a working installation of NumPy, the easiest way to install lm-sim is using pip:
```
pip install lm-sim
```

### Example Usage 
A simple example to compute probabilistic error consistency $k_p$ in the context of MCQs. Input has be to formatted as follows:
- `prob_a`: list[np.array], containing the softmax output probabilties of model a
- `prob_b`: list[np.array], containing the softmax output probabilties of model b
- `gt`: list[int], containing the index of the ground truth 

```
from lmsim.metrics import K_p

calculator = K_p()
calculator.compute_kp(prob_a, prob_b, gt)

```