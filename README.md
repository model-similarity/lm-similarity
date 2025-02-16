# LM-Similarity

**lm-sim** is a Python module for computing similarity between Language Models and is distributed under the MIT license. <br>
For a detailed discussion on metrics and applications to AI oversight, see [our paper](https://arxiv.org/abs/2502.04313)


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
- CAPA (Chance Adjusted Probabilistic Agreement), $\kappa_p$ (default)
- CAPA (Chance Adjusted Probabilistic Agreement), $\kappa_p$ discrete
- Error Consistency

#### Compute similarity based on CAPA, $\kappa_p$

Below is a simple example on how to compute similarity between 2 models based on $k_p$. The input has be to formatted as follows:
- `output_a`: list[np.array], containing the softmax output probabilties of model a
- `output_b`: list[np.array], containing the softmax output probabilties of model b
- `gt`: list[int], containing the index of the ground truth 

```
from lmsim.metrics import CAPA

capa= CAPA()
capa.compute_k(output_a, output_b, gt)

```

For a discrete computation (when output probabilities are not availble) set the flag `prob=False` and the input must be formatted as one-hot vectors:
- `output_a`: list[np.array], one-hot vector of model a
- `output_b`: list[np.array], one-hot vector of model b

```
from lmsim.metrics import CAPA

capa = CAPA(prob=False)
capa.compute_k(output_a, output_b, gt)
```

#### Compute similarity based on Error Consistency
```
from lmsim.metrics import EC

ec = EC()
ec.compute_k(output_a, output_b, gt)
```
Implementation supports both softmax output probabilties or one-hot vector as input.

#### Citation
To cite our work:
```
@misc{goel2025greatmodelsthinkalike,
      title={Great Models Think Alike and this Undermines AI Oversight}, 
      author={Shashwat Goel and Joschka Struber and Ilze Amanda Auzina and Karuna K Chandra and Ponnurangam Kumaraguru and Douwe Kiela and Ameya Prabhu and Matthias Bethge and Jonas Geiping},
      year={2025},
      eprint={2502.04313},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.04313}, 
}
```

