# import necessary packages
import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams

# Define the path to read evaluation stat files and output final results
result_stat_path = '../../results/output stats'
stat_file_list = os.listdir(result_stat_path)

RESULTS_ANALYSIS_OUTPUT_PATH = '../../results/results analysis/'
results = []
result_csv_path = RESULTS_ANALYSIS_OUTPUT_PATH + "model_eval_results.csv"
existing_model_names = set()

# Load existing results if the CSV already exists
if os.path.exists(result_csv_path):
    existing_df = pd.read_csv(result_csv_path)
    existing_model_names = set(existing_df['model_name'].tolist())

# Loop through each stat file and compute metrics
for file in stat_file_list:
    model_name = file.split('-cuad-output-stat.csv')[0]
    # Skip already processed models
    if model_name in existing_model_names:
        print(f"Skipping {model_name} (already in results CSV)")
        continue
    
    # Load labels for comparison later
    output_df = pd.read_csv(f'../../results/complete outputs/{model_name}-cuad-output.csv')
    labels = output_df['labels'].to_list()

    stat_df = pd.read_csv(f'{result_stat_path}/{file}', quoting=csv.QUOTE_MINIMAL)
    output_df = pd.read_csv(f'../../results/complete outputs/{model_name}-cuad-output.csv', quoting=csv.QUOTE_ALL)
    print(model_name, 'Valid data cnt: ', len(output_df))
    
    outputs = output_df['outputs'].to_list()
    labels = output_df['labels'].to_list()
    check_include = stat_df['classification'].to_list()[:len(outputs)]

    tp, tn, fn, fp, jac_tp, jac_fn = 0, 0, 0, 0, 0, 0
    no_related_clause_cnt, false_no_related_clause_cnt = 0, 0

    # Count 'no related clause' appearances and false rate 
    for output, label in zip(outputs, labels):
        label = eval(label)
        if isinstance(output, str) and 'no related clause' in output.strip(" \n`").lower():
            no_related_clause_cnt+=1
            if len(label) != 0:
                false_no_related_clause_cnt +=1


    no_related_clause_rate = no_related_clause_cnt/len(output_df)
    false_no_related_clause_rate = false_no_related_clause_cnt/1244
    
     # Calculate confusion matrix values for correctness
    for classification, output, label in zip(check_include, outputs, labels):
        label = eval(label)
        if len(label) == 0:
            if isinstance(output, str) and 'no related clause' in output.strip(" \n`").lower():
                tn += 1
            else:
                fp += 1
        else:
            if classification:
                tp += 1
            else:
                fn += 1
                
    acc = (tp + tn) / (tp + tn + fn + fp) if (tp + tn + fn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2*precision*recall)/(precision+recall)
    f2_score = (5*precision*recall) / (4*precision+recall)

    print(f'Acc: {acc:.3f}; Precision: {precision:.3f}; Recall: {recall:.3f}; F1_score: {f1_score:.3f}; F2_score: {f2_score:.3f}; no related clause rate: {no_related_clause_rate:.3f}; false no related clause rate: {false_no_related_clause_rate:.3f}')
    print()

    results.append({
    "model_name": model_name,
    "valid_data_count": len(output_df),
    "accuracy": round(acc, 3),
    "precision": round(precision, 3),
    "recall": round(recall, 3),
    "f1_score": round(f1_score, 3),
    "f2_score": round(f2_score, 3),
    "no related clause count": round(no_related_clause_rate, 3),
    "false no related clause rate": round(false_no_related_clause_rate, 3)
    })

results_df = pd.DataFrame(results)

# Append new results to existing CSV (or create if missing)
if os.path.exists(result_csv_path):
    results_df.to_csv(result_csv_path, mode='a', index=False, header=False)
else:
    results_df.to_csv(result_csv_path, index=False)


results_df = pd.read_csv(result_csv_path)

# define the function to calculate jaccard similarity
def get_jaccard(gt, pred):
    remove_tokens = [".", ",", ";", ":"]
    for token in remove_tokens:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower()
    pred = pred.lower()
    gt = gt.replace("/", " ")
    pred = pred.replace("/", " ")

    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))

    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard

jac_sim_dict = {}

# Define the path to read complete output files
complete_outputs_path = '../../results/complete outputs'
output_file_list = os.listdir(complete_outputs_path)

# Loop through each complete output file and compute jaccard similarity coefficients
for file in output_file_list:
    model_name = file.split('-cuad-output.csv')[0]
    output_df = pd.read_csv(f'../../results/complete outputs/{model_name}-cuad-output.csv', quoting=csv.QUOTE_ALL)
    if len(output_df) < 4000 or 'Qwen2.5' in model_name:
        continue
    
    outputs = output_df['outputs'].to_list()
    labels = output_df['labels'].to_list()

    jac_sim = []
    for output, label in zip(outputs, labels):
        label = eval(label)
        if len(label) == 0:
            continue
        else:
            jac_sim.append(get_jaccard(' '.join(label), output.strip(" \n`")))
    
    jac_sim_dict[model_name] = jac_sim



# Define the function to calcuate statistics for jaccard similarity
def compute_stats(jac_sim_dict):
    summary_data = []

    for model, sims in jac_sim_dict.items():
        sims = np.array(sims)
        summary_data.append({
            'model': model,
            'mean': round(np.mean(sims), 3),
            'median': round(np.median(sims), 3),
            'std': round(np.std(sims), 3),
            'p25': round(np.percentile(sims, 25), 3),
            'p75': round(np.percentile(sims, 75), 3),
            'min': round(np.min(sims), 3),
            'max': round(np.max(sims), 3),
        })

    return pd.DataFrame(summary_data).set_index('model')

jaccard_summary_df = compute_stats(jac_sim_dict)
jaccard_summary_df = jaccard_summary_df.sort_values(by='mean', ascending=False)

pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.expand_frame_repr', False) # Do NOT wrap to new line
pd.set_option('display.width', 200)               # Wider layout
print(jaccard_summary_df)



# First, make sure df2 has its index as a column
jaccard_summary_df = jaccard_summary_df.reset_index()

# Rename columns
jaccard_summary_df = jaccard_summary_df.rename(columns={"model": "model_name", "mean": "jaccard_similarity_mean"})

# Merge with df1 on model_name
merged_df = results_df.merge(jaccard_summary_df[["model_name", "jaccard_similarity_mean"]], on="model_name", how="left")

result_csv_path = RESULTS_ANALYSIS_OUTPUT_PATH + "model_eval_results_v2.csv" 
merged_df.to_csv(result_csv_path, index=False)




# Code to generate figures:

# # Figure: False Rate

# Ensure Latex binaries are available
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

# Update Matplotlib's default rendering settings for consistent figure styling
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # Computer Modern
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
})

# Sort the dataframe by the rate column
sorted_fn_df = merged_df.sort_values(by="false no related clause rate", ascending=True)

color = '#A8DADC'    # (Light Teal)

# Plotting
plt.figure(figsize=(7, 5))
plt.bar([model_name_map[model] for model in sorted_fn_df["model_name"]], sorted_fn_df["false no related clause rate"], color=color)
plt.xticks(rotation=45, ha='right')
plt.ylabel("False Rate", fontweight='bold', fontsize=13)
plt.xlabel("Model Name", fontweight='bold', fontsize=13)
# plt.title("False 'No Related Clause' Rate by Model")
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(RESULTS_ANALYSIS_OUTPUT_PATH + 'fnr_comparison_v2.pdf', dpi=300, bbox_inches='tight')
plt.show()


# # Figure: Qwen3 thinking vs non_thinking


Qwen_results_df = merged_df[merged_df['model_name'].str.startswith('Qwen3')]
# Rename each model
model_group_map = {
    "Qwen3-4B": "Qwen3 4B",
    "Qwen3-8B": "Qwen3 8B",
    "Qwen3-14B": "Qwen3 14B",
    "Qwen3-8B-AWQ": "Qwen3 8B AWQ",
    "Qwen3-8B-FP": "Qwen3 8B FP"
}






import os
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # Computer Modern
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
})

# Extract thinking / non_thinking
Qwen_results_df['type'] = Qwen_results_df['model_name'].apply(
    lambda x: 'non_thinking' if 'non_thinking' in x else 'thinking'
)

# Extract model group
Qwen_results_df['group'] = Qwen_results_df['model_name'].str.extract(
    r'(Qwen3-\d+B(?:-[A-Z]+)?)'
)

# Define group order
group_order = (
    Qwen_results_df[Qwen_results_df['group'] != 'Qwen3-14B']
    .drop_duplicates(subset='group')
    .sort_values('model_name')['group']
    .tolist()
)
if 'Qwen3-14B' in Qwen_results_df['group'].values:
    group_order.append('Qwen3-14B')

Qwen_results_df["group"] = Qwen_results_df["group"].map(model_group_map)
group_order = [model_group_map[group] for group in group_order]

# Pivot data for F1 and Jaccard
pivot_f1 = Qwen_results_df.pivot_table(index='group', columns='type', values='f1_score')
pivot_jac = Qwen_results_df.pivot_table(index='group', columns='type', values='jaccard_similarity_mean')

# Reindex for consistent order
pivot_f1 = pivot_f1.reindex(group_order)
pivot_jac = pivot_jac.reindex(group_order)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'hspace': 0.1})
color_map = {
    'non_thinking': '#A8DADC',  # (Light Teal)
    'thinking': '#F4A261',      # (Soft Coral/Apricot)
}

# F1 Score Bar Plot
pivot_f1.plot(kind='bar', ax=ax1, color=[color_map.get(col, '#999999') for col in pivot_f1.columns])
ax1.set_ylabel('F1 Score', fontweight='bold', fontsize=13)
ax1.set_ylim(0, 0.6)
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend(
    fontsize=11,           # smaller font
    loc='upper left',
    ncol=2,                # ← Put legend in 1 row (2 columns)
    frameon=False          # Optional: remove box around legend
)
ax1.grid(axis='y', linestyle='--', linewidth=0.5)

# Jaccard Similarity Bar Plot
pivot_jac.plot(kind='bar', ax=ax2, color=[color_map.get(col, '#999999') for col in pivot_jac.columns])
ax2.set_ylabel('Average Jaccard Similarity', fontweight='bold', fontsize=13)
ax2.set_ylim(0, 0.5)
ax2.set_yticks(np.arange(0, 0.6, 0.1))
ax2.set_xlabel('Model Group', fontweight='bold', fontsize=13)
ax2.legend().set_visible(False)  # Hide duplicate legend
ax2.grid(axis='y', linestyle='--', linewidth=0.5)

# Rotate x ticks
plt.xticks(rotation=45)

# Save and show
plt.tight_layout()
plt.savefig(RESULTS_ANALYSIS_OUTPUT_PATH + 'qwen3_f1_jaccard_stacked_bars.pdf', dpi=300, bbox_inches='tight')
plt.show()


# # Figure: Performance by question category
# model 1: gpt-4.1-mini
gpt_output_df = pd.read_csv(f'../../results/complete outputs/gpt-4.1-mini-2025-04-14-cuad-output.csv')

gpt_categories_lst = gpt_output_df['questions'].to_list()
gpt_outputs = gpt_output_df['outputs'].to_list()
gpt_labels = gpt_output_df['labels'].to_list()
gpt_check_include = gpt_output_df['classification'].to_list()[:len(outputs)]


question_metrics = {}
gpt_q_results = []

# calculate metrics by question category
for output, label, classification, question in zip(gpt_outputs, gpt_labels, gpt_check_include, gpt_categories_lst):
    label = eval(label)
    q_clean = question.strip()  # full question text

    if q_clean not in question_metrics:
        question_metrics[q_clean] = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    if len(label) == 0:
        if isinstance(output, str) and 'no related clause' in output.strip(" \n`").lower():
            question_metrics[q_clean]['TN'] += 1
        else:
            question_metrics[q_clean]['FP'] += 1
    else:
        if classification:
            question_metrics[q_clean]['TP'] += 1
        else:
            question_metrics[q_clean]['FN'] += 1

    # Compute metrics per unique question
for question, counts in question_metrics.items():
    TP = counts['TP']
    TN = counts['TN']
    FP = counts['FP']
    FN = counts['FN']
    total = TP + TN + FP + FN

    q_acc = (TP + TN) / total if total > 0 else 0
    q_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    q_rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    q_f1 = (2*q_prec*q_rec)/(q_prec+q_rec) if (q_prec+q_rec)>0 else 0
    q_f2 = (5 * q_prec * q_rec) / (4 * q_prec + q_rec) if (4 * q_prec + q_rec) > 0 else 0

    print(f'[Question] {question[:50]}... → Acc: {q_acc:.3f}; Prec: {q_prec:.3f}; Rec: {q_rec:.3f}; F1: {q_f1:.3f}; F2: {q_f2:.3f}')
    
    gpt_q_results.append({
    "model_name": "gpt-4.1-mini",
    "question": question,
    "accuracy": round(q_acc, 3),
    "precision": round(q_prec, 3),
    "recall": round(q_rec, 3),
    "f1_score": round(q_f1, 3),
    "f2_score": round(q_f2, 3),
    })


gpt_q_results_df = pd.DataFrame(gpt_q_results)

# repeat same process for gemma 3 12b model
gemma_output_df = pd.read_csv(f'../../results/complete outputs/gemma-3-12b-it-cuad-output.csv')

gemma_categories_lst = gemma_output_df['questions'].to_list()
gemma_outputs = gemma_output_df['outputs'].to_list()
gemma_labels = gemma_output_df['labels'].to_list()
gemma_check_include = gemma_output_df['classification'].to_list()[:len(outputs)]

question_metrics = {}
gemma_q_results = []

for output, label, classification, question in zip(gemma_outputs, gemma_labels, gemma_check_include, gemma_categories_lst):
    label = eval(label)
    q_clean = question.strip()  # full question text

    if q_clean not in question_metrics:
        question_metrics[q_clean] = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    if len(label) == 0:
        if isinstance(output, str) and 'no related clause' in output.strip(" \n`").lower():
            question_metrics[q_clean]['TN'] += 1
        else:
            question_metrics[q_clean]['FP'] += 1
    else:
        if classification:
            question_metrics[q_clean]['TP'] += 1
        else:
            question_metrics[q_clean]['FN'] += 1

    # Compute metrics per unique question
for question, counts in question_metrics.items():
    TP = counts['TP']
    TN = counts['TN']
    FP = counts['FP']
    FN = counts['FN']
    total = TP + TN + FP + FN

    q_acc = (TP + TN) / total if total > 0 else 0
    q_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    q_rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    q_f1 = (2*q_prec*q_rec)/(q_prec+q_rec) if (q_prec+q_rec)>0 else 0
    q_f2 = (5 * q_prec * q_rec) / (4 * q_prec + q_rec) if (4 * q_prec + q_rec) > 0 else 0

    print(f'[Question] {question[:50]}... → Acc: {q_acc:.3f}; Prec: {q_prec:.3f}; Rec: {q_rec:.3f}; F1: {q_f1:.3f}; F2: {q_f2:.3f}')
    
    gemma_q_results.append({
    "model_name": "gemma-3-12b-it",
    "question": question,
    "accuracy": round(q_acc, 3),
    "precision": round(q_prec, 3),
    "recall": round(q_rec, 3),
    "f1_score": round(q_f1, 3),
    "f2_score": round(q_f2, 3),
    })

gemma_q_results_df = pd.DataFrame(gemma_q_results)

# get jaccard similarity scores by category for gpt and gemma
gpt_output_df = pd.read_csv(f'../../results/complete outputs/gpt-4.1-mini-2025-04-14-cuad-output.csv')

gpt_categories_lst = gpt_output_df['questions'].to_list()
gpt_outputs = gpt_output_df['outputs'].to_list()
gpt_labels = gpt_output_df['labels'].to_list()

question_metrics = {}
gpt_q_results = []
gpt_jac_sim = []

for output, label, question in zip(gpt_outputs, gpt_labels, gpt_categories_lst):
    label = eval(label)
    if len(label) == 0:
        continue
    else:
        gpt_jac_sim.append({
            "model_name": "gpt",
            "question": question,
            "jaccard": get_jaccard(' '.join(label), output.strip(" \n`")),
        })

gemma_output_df = pd.read_csv(f'../../results/complete outputs/gemma-3-12b-it-cuad-output.csv')

gemma_categories_lst = gemma_output_df['questions'].to_list()
gemma_outputs = gemma_output_df['outputs'].to_list()
gemma_labels = gemma_output_df['labels'].to_list()

gemma_jac_sim = []

for output, label, question in zip(gemma_outputs, gemma_labels, gemma_categories_lst):
    label = eval(label)
    if len(label) == 0:
        continue
    else:
        gemma_jac_sim.append({
            "model_name": "gemma",
            "question": question,
            "jaccard": get_jaccard(' '.join(label), output.strip(" \n`")),
        })

gpt_jac_sim_df = pd.DataFrame(gpt_jac_sim)
gemma_jac_sim_df = pd.DataFrame(gemma_jac_sim)

gpt_jac_sim_grouped_df = gpt_jac_sim_df.groupby('question', as_index=False)['jaccard'].mean()
gemma_jac_sim_grouped_df = gemma_jac_sim_df.groupby('question', as_index=False)['jaccard'].mean()


# Merge the needed GPT metrics into one DataFrame
merged_sorted_df = gpt_q_results_df[['question', 'f1_score']].merge(
    gpt_jac_sim_grouped_df[['question', 'jaccard']],
    on='question',
    how='left'
)

# Sort only by GPT F1 score descending
merged_sorted_df = merged_sorted_df.sort_values(
    by='f1_score',
    ascending=False
)

# Use the sorted question order
sorted_questions = merged_sorted_df['question']

# Reindex other GPT and Gemma dataframes
gpt_sorted_f1_df = gpt_q_results_df.set_index('question').reindex(sorted_questions).reset_index()
gemma_sorted_f1_df = gemma_q_results_df.set_index('question').reindex(sorted_questions).reset_index()

gpt_sorted_jac_df = gpt_jac_sim_grouped_df.set_index('question').reindex(sorted_questions).reset_index()
gemma_sorted_jac_df = gemma_jac_sim_grouped_df.set_index('question').reindex(sorted_questions).reset_index()




os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # Computer Modern
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
})

gpt_color = '#A8DADC'    # (Light Teal)
gemma_color = '#F4A261'  # (Soft Coral/Apricot)

x = np.arange(len(sorted_questions))
width = 0.35

questions = gpt_sorted_f1_df['question']

# F1 Score values
gpt_f1_scores = gpt_sorted_f1_df['f1_score']
gemma_f1_scores = gemma_sorted_f1_df['f1_score']

# Jaccard values
gpt_jaccard = gpt_sorted_jac_df['jaccard']
gemma_jaccard = gemma_sorted_jac_df['jaccard']

# Create vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

x_f1 = np.arange(len(gpt_sorted_f1_df))    # For first plot
x_jac = np.arange(len(gpt_sorted_jac_df))  # For second plot

# --- F1 Score subplot ---
ax1.bar(x_f1 - width/2, gpt_f1_scores, width=width, label='GPT 4.1 mini', color=gpt_color)
ax1.bar(x_f1 + width/2, gemma_f1_scores, width=width, label='Gemma 3 12B', color=gemma_color)
ax1.set_xticks(x_f1)
ax1.set_xticklabels(gpt_sorted_f1_df['question'], rotation=45, ha='right')
ax1.set_ylabel('F1 Score', fontweight='bold', fontsize=15)
ax1.set_yticks(np.arange(0, 1.1, 0.2))
# ax1.set_title('F1 Score by Question Category')
ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
ax1.legend(
    fontsize=12,           # smaller font
    loc='upper right',
    ncol=2,                # ← Put legend in 1 row (2 columns)
    frameon=False          # Optional: remove box around legend
)

# --- Jaccard Similarity subplot ---
ax2.bar(x_jac - width/2, gpt_jaccard, width=width, label='GPT 4.1 mini', color=gpt_color)
ax2.bar(x_jac + width/2, gemma_jaccard, width=width, label='Gemma 3 12B', color=gemma_color)
ax2.set_xticks(x_jac)
ax2.set_xticklabels(gpt_sorted_jac_df['question'], rotation=45, ha='right')
ax2.set_ylabel('Average Jaccard Similarity', fontweight='bold', fontsize=15)
ax2.set_yticks(np.arange(0, 1.1, 0.2))
# ax2.set_title('Jaccard Similarity by Question Category')
ax2.set_xlabel('Question Category', fontweight='bold', fontsize=15)
ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
ax2.legend().set_visible(False)  # Hide duplicate legend

plt.tight_layout()
plt.savefig(RESULTS_ANALYSIS_OUTPUT_PATH + 'gpt_vs_gamma_f1_jaccard_vertical_by_category.pdf', bbox_inches='tight')
plt.show()


