# ContractEval
This repository includes the implementation for paper "ContractEval: Benchmarking LLMs for Clause-Level Legal Risk Identification in Commercial Contracts".

We welcome any questions and requests regarding code implementation or any other issues.

## Environment
Python 3.9.21

Necessary packages: see the [requirments.txt](requirements.txt) for detailed information.


## Dataset description
We leveraged the test data of [CUAD](https://github.com/TheAtticusProject/cuad) projects to implement ContractEval. 

## Implementation

![Workflow Diagram](/diagram.png)

ContractEval evaluates 4 proprietary LLMs and 15 open-source models on contract review task. We design system prompt asking LLMs to act as a legal assistant to extract and return only the sentence(s) from the Context that directly address or relate to the questions. If no part of the Context is relevant to the Question, LLMs should respond with: "No related clause."

## Example code for proprietary models
First, change the value of `api_key`, `model_name`, `model_id` in Line 30, 31, 83 in `proprietary_model.py` prespectively.

Then, the template code for proprietary models is run by:

```
python proprietary_model.py
```

If you are using Python 3 specifically:
```
python3 proprietary_model.py
```

## Example code for open-source models
First, change the value of `model_name` in Line 22 in `open_source_model.py`.

The template code for open-source models is run by:

```
python open_source_model.py
```

If you are using Python 3 specifically:
```
python3 open_source_model.py
```

## Example code for evaluatio
We evaluate the models' performance from three persepctives: 1. correctness (F1 and F2 scores); 2. Output Effectiveness (Jaccard similarity coefficients); 3. laziness (rate of falsely outputting ``no related clauses"). 

The example code for evaluation metrics is run by:
```
python Evaluation.py
```

If you are using Python 3 specifically:
```
python3 Evaluation.py
```







