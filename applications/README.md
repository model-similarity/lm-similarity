# Applications 

We use $\kappa_p$ to showcase the importance of reporting and correcting for model similarity, especially in the emerging paradigm of AI oversight. For a detailed explanation please read ["Great Models Think Alike this Undermines AI Oversight" (Goel at al., 2025)](http://example.com). As summarised in Fig.1 we explore the following applications:


<div style="display: flex; align-items: flex-start;">
    <div style="margin-right: 1rem;">
        <img src="./images/contributions.png" alt="Our Main Contributions" width="400"/>
        <br/>
        <em>Figure 1: Our Main Contributions</em>
        <br/><br/>
    </div>
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