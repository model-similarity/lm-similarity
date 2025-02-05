# Applications 

We use $\kappa_p$ to showcase the importance of reporting and correcting for model similarity, especially in the emerging paradigm of AI oversight. For a detailed explanation please read ["Great Models Think Alike this Undermines AI Oversight" (Goel at al., 2025)](http://example.com). As summarised in Fig.1 we explore the following applications:

<p align="center">
  <img src="./images/contributions.png" alt="Our Main Contributions" width="400">
  <br/>
     <em>Figure 1: Our Main Contributions</em>
  <br/><br/>
</p>

<div style="flex: 1;">
        "Great Models Think Alike":
        <ul>
            <li>Model Capabilities</li>
        </ul>
       "this Undermines AI Oversight":
        <ul>
            <li>LM-as-a-Judge</li>
            <li>LM-as-a-Annotator</li>
        </ul>
</div>


## Results
We find the following observations with respect to the applications listed above:
- **Model Capabilities**: model errors are getting more correlated as capabilities increase
- **LM-as-a-Judge**: LLM-as-a-judge scores are biased towards more similar
models controlling for the model’s capability
- **LM-as-a-Annotator**: gain from training strong models on annotations of weak supervisors
(weak-to-strong generalization) is higher when the two models are more different 


## Experiments 
Code files and instructions on reproducing experiments and downloading data for each application are provided in separate folders as follows:

- **Model Capabilities**: $\rightarrow$ Capabilities 
- **LM-as-a-Judge** $\rightarrow$ Judges
- **LM-as-a-Annotator** $\rightarrow$ Annotators

## Set Up the Environment

Create a new environment:

```
conda create -n env_sim python=3.11 -y
conda activate env_sim
```

Install torch:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install a CUDA version that exactly matches the pytorch-cuda version. In our environment, this was necessary to install flash attention.

```
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0
```

Install the remaining requirements using pip.

```
pip install -r requirements.txt
```
