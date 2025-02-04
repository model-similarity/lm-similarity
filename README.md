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
Currently we support the calcualtion of 3 similarity metrics in the context of MCQ datasets: 
- Goels $k$ (discrete)
- Goels $k_p$ (probabilistic) 
- Error Consistency

#### Goels $k$ and Goels $k_p$

Below is a simple example on how to compute Goels $k_p$. The input has be to formatted as follows:
- `output_a`: list[np.array], containing the softmax output probabilties of model a
- `output_b`: list[np.array], containing the softmax output probabilties of model b
- `gt`: list[int], containing the index of the ground truth 

```
from lmsim.metrics import Goels_k

goels_k = Goels_k()
goels_k.compute_k(output_a, output_b, gt)

```

For a discrete computation (when output probabilities are not availble) set the flag `prob=False` and the input must be formatted as one-hot vectors:
- `output_a`: list[np.array], one-hot vector of model a
- `output_b`: list[np.array], one-hot vector of model b

```
from lmsim.metrics import Goels_k

goels_k = Goels_k(prob=False)
goels_k.compute_k(output_a, output_b, gt)
```

#### Error Consistency
```
from lmsim.metrics import EC

ec = EC()
ec.compute_k(output_a, output_b, gt)
```
Implementation supports both softmax output probabilties or one-hot vector as input.
