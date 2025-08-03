# import necessary packages
import os
import torch
from transformers import pipeline
import transformers
import datasets
import csv
import pandas as pd
from tqdm import tqdm
import re

import numpy as np
import matplotlib.pyplot as plt

# Set environment variable 
os.environ['TORCH_LOGS'] = "recompiles"

torch._dynamo.config.cache_size_limit = 10000
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Define the model name
model_name = 'to_insert_model_name' # insert the model name

# Use a pipeline as a high-level helper
transformers.logging.set_verbosity_error()
pipe = pipeline("text-generation", model=model_name, model_kwargs={'cache_dir': './cache_dir'}, device_map='auto')

# Load the CUAD dataset
cuad_ds = datasets.load_dataset('theatticusproject/cuad-qa', cache_dir='./cache_dir', trust_remote_code=True)
train_ds = cuad_ds['train']
test_ds = cuad_ds['test']

# Define prompt template
prompt = """Context: 
```
{context}
```
Question:
```
{question}
```
"""

# System prompt
system_prompt = """You are an assistant with strong legal knowledge, supporting senior lawyers by preparing reference materials.
Given a Context and a Question, extract and return only the sentence(s) from the Context that directly address or relate to the Question.
Do not rephrase or summarize in any way—respond with exact sentences from the Context relevant to the Question. If a relevant sentence contains unrelated elements such as page numbers or whitespace, include them exactly as they appear.
If no part of the Context is relevant to the Question, respond with: "No related clause."
"""


# Function to run the pipeline with formated prompt structure
def pipeline_inference(pipeline, prompt, data):
    user_input = prompt.format(context=data['context'], question=data['question'])
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    output = pipe(conversation, do_sample=False, max_new_tokens=5000, temperature=0.)
    return output[0]['generated_text'][-1]['content']


# Load previously saved outputs/statistics if available
file_name = f'./{model_name.split("/")[-1]}-cuad-output.csv'
if os.path.isfile(file_name):
    test_df = pd.read_csv(f'./{model_name.split("/")[-1]}-cuad-output-stat.csv', quoting=csv.QUOTE_MINIMAL)
    output_df = pd.read_csv(f'./{model_name.split("/")[-1]}-cuad-output.csv', quoting=csv.QUOTE_ALL)
    ids = test_df['id'].to_list()
    questions = test_df['questions'].to_list()
    check_include = test_df['classification'].to_list()
    predicts = test_df['predict'].to_list()
    label_binary = test_df['label'].to_list()
    output_lens = test_df['output_char'].to_list()
    label_lens = test_df['label_char'].to_list()

    outputs = output_df['outputs'].to_list()
    labels = output_df['labels'].to_list()
else:
    ids, questions, outputs, labels = [], [], [], []
    check_include, output_lens, label_lens = [], [], []
    predicts, label_binary = [], []


# Iterate through test set and perform inference + evaluation
for i, data in enumerate(tqdm(test_ds)):
    if data['id'] in ids:
        continue
    torch.cuda.empty_cache()
    try:
        res = pipeline_inference(pipe, prompt, data)
    except torch.cuda.OutOfMemoryError as e:
        print(f"{data['id']}: CUDA out-of-memory error caught!")
        torch.cuda.empty_cache()
        continue
    question = re.search(r'"(.*?)"', data['question']).group(1)
    ids.append(data['id'])
    questions.append(question)
    outputs.append(res)
    labels.append(data['answers']['text'])

    output = res
    label = data['answers']['text']
    if len(label) == 0:
        check_include.append('no related clause' in output.lower())
    else:
        check_include.append(all(substr.strip(" \n`") in output.strip(" \n`") for substr in label))
    output_lens.append(len(output.strip(" \n`")))
    label_lens.append(sum(len(substr.strip(" \n`")) for substr in label))
    predicts.append(not 'no related clause' in output.lower())
    label_binary.append(len(label) > 0)
    
    # Save detailed outputs and statistics
    result_df = pd.DataFrame({
        'id': ids,
        'questions': questions,
        'outputs': outputs,
        'labels': labels,
        'classification': check_include,
        'output_char': output_lens,
        'label_char': label_lens,
    })
    result_df.to_csv(f'./{model_name.split("/")[-1]}-cuad-output.csv', index=False, quoting=csv.QUOTE_ALL)

    result_df = pd.DataFrame({
        'id': ids,
        'questions': questions,
        'classification': check_include,
        'predict': predicts,
        'label': label_binary,
        'output_char': output_lens,
        'label_char': label_lens,
    })
    result_df.to_csv(f'./{model_name.split("/")[-1]}-cuad-output-stat.csv', index=False, quoting=csv.QUOTE_MINIMAL)
