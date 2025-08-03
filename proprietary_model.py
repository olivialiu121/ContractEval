# import necessary packages
import datasets
from openai import OpenAI #e.g. Import OpenAI client
import time
from tqdm import tqdm
import re
import csv
import pandas as pd


# Load the CUAD dataset
cuad_ds = datasets.load_dataset('theatticusproject/cuad-qa', cache_dir='./cache_dir', trust_remote_code=True)
train_ds = cuad_ds['train']
test_ds = cuad_ds['test']
train_ds


# Define prompt structure
prompt = """Context: 
```
{context}
```
Question:
```
{question}
```
"""

# E.g. set up OpenAI client
client = OpenAI(api_key='insert_your_api_key')
model_name = 'insert_model_name'

# Define function to send a system/user prompt to GPT models and return the generated response/output
def invoke_gpt(system_prompt, user_prompt, model_id="insert_model_id", temperature=0., max_token=100000):
    t0 = time.time()
    response = client.responses.create(
        model=model_id,
        input=[
        {
          "role": "system",
          "content": [
            {
              "type": "input_text",
              "text": system_prompt
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": user_prompt # prompt.replace('{question}', pos_data['question']).replace("{context}", pos_data['context'])
            }
          ]
        },
        ],
        text={
            "format": {
              "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=temperature,
        max_output_tokens=max_token,
        top_p=0.9,
    )
    t1 = time.time()
    # print('Inference time: ', t1 - t0)
    return response.output[0].content[0].text


# Define system prompt: instructing models to only return relevant sentences from context
system_prompt = """You are an assistant with strong legal knowledge, supporting senior lawyers by preparing reference materials.
Given a Context and a Question, extract and return only the sentence(s) from the Context that directly address or relate to the Question.
Do not rephrase or summarize in any way—respond with exact sentences from the Context relevant to the Question. If a relevant sentence contains unrelated elements such as page numbers or whitespace, include them exactly as they appear.
If no part of the Context is relevant to the Question, respond with: "No related clause."
"""

# Initialize lists to store evaluation data
ids, questions, outputs, labels = [], [], [], []
model_id = "insert_model_id"

# Iterate through the test set and run GPT inference on each sample
for data in tqdm(test_ds):
    user_input = prompt.format(context=data['context'], question=data['question'])
    res = invoke_gpt(system_prompt, user_input, model_id=model_id, temperature=0., max_token=400000)
    question = re.search(r'"(.*?)"', data['question']).group(1)
    ids.append(data['id'])
    questions.append(question)
    outputs.append(res)
    labels.append(data['answers']['text'])

# Evaluate model outputs using inclusion
check_include = []
output_lens, label_lens = [], []
for output, label in zip(outputs, labels):
    if len(label) == 0:
        check_include.append(output.strip(" \n`").lower().startswith('no related clause'))
    else:
        check_include.append(all(substr.strip(" \n`") in output.strip(" \n`") for substr in label))
    output_lens.append(len(output.strip(" \n`")))
    label_lens.append(sum(len(substr.strip(" \n`")) for substr in label))



# Save full outputs to CSV for further analysis
result_df = pd.DataFrame({
    'id' : ids,
    'questions': questions,
    'outputs': outputs,
    'labels': labels,
    'classification': check_include,
    'output_char': output_lens,
    'label_char': label_lens,
})
result_df.to_csv(f'./{model_name.split("/")[-1]}-cuad-output.csv', index=False, quoting=csv.QUOTE_ALL)

# Save statistics (a lighter version of results) for performance metrics
result_df = pd.DataFrame({
    'id' : ids,
    'questions': questions,
    'classification': check_include,
    'predict': [not output.strip(" \n`").lower().startswith('no related clause') for output in outputs],
    'label': [len(label) > 0 for label in labels],
    'output_char': output_lens,
    'label_char': label_lens,
})
result_df.to_csv(f'./{model_name.split("/")[-1]}-cuad-output-stat.csv', index=False, quoting=csv.QUOTE_MINIMAL)








