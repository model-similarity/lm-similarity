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
A simple example to compute Goels $k$ in the context of MCQs (default computation is probabilistic Goels $k_p$). Input has be to formatted as follows:
- `output_a`: list[np.array], containing the softmax output probabilties of model a
- `output_b`: list[np.array], containing the softmax output probabilties of model b
- `gt`: list[int], containing the index of the ground truth 

```
from lmsim.metrics import Goels_k

calculator = Goels_k()
calculator.compute_k(output_a, output_b, gt)

```

For a discrete computation, when output probabilities are not availble, set the flag `prob=False` and the input must be formatted as one-hot vectors:
- `output_a`: list[np.array], one-hot vector of model a
- `output_b`: list[np.array], one-hot vector of model b

```
from lmsim.metrics import Goels_k

calculator = Goels_k(prob=False)
calculator.compute_k(output_a, output_b, gt)
```