# %%
import os
import json
import re
import pandas as pd
import numpy as np
import csv
import torch
import jionlp as jio
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau 

# %%
#create a new dataframe empty 
month_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
# for month in month_list:
#     with open(f'dataset/dataset_training/2023{month}.csv',encoding="utf-8-sig") as f:
with open('dataset/dataset_training/202312.csv',encoding="utf-8-sig") as f:
    data = pd.read_csv(f)
    data.head()


# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
# load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./m2m100_418M")
tokenizer = AutoTokenizer.from_pretrained("./m2m100_418M")


# %%
#column p_claim is the answer  and column p_fact is the question
data = data[['p_claim', 'p_fact']]
data = data.rename(columns={'p_claim': 'answer', 'p_fact': 'question'})
data = data.dropna()
data = data.reset_index(drop=True)
data.head()
 
data['question'] = data['question'].apply(lambda x: jio.clean_text(x))
data['answer'] = data['answer'].apply(lambda x: jio.clean_text(x))
data['question'] = data['question'].apply(lambda x: re.sub(r'\s+', ' ', x))
data['answer'] = data['answer'].apply(lambda x: re.sub(r'\s+', ' ', x))

# %%
#split the data into training and testing
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# %%
def preporcessing(data):
    #tokenize the data
    inputs = tokenizer(data['question'].tolist(), return_tensors='pt', padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(data['answer'].tolist(), return_tensors='pt', padding="max_length", truncation=True, max_length=128,)
    return inputs, targets

train_inputs, train_targets = preporcessing(train)
test_inputs, test_targets = preporcessing(test)

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        labels = self.targets['input_ids'][idx]
        labels[labels == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
train_dataset = CustomDataset(train_inputs, train_targets)
test_dataset = CustomDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=1e-5,
    
    

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

device = torch.device('cuda')
model.to(device)
trainer.train()

trainer.evaluate()

# %%
#BLEU score
from datasets import load_metric
metric = load_metric("sacrebleu")
model.eval()
predictions = []
labels = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels_ids = batch['labels'].to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    labels.extend(tokenizer.batch_decode(labels_ids, skip_special_tokens=True))

metric.compute(predictions=[predictions], references=[[labels]])



# %%
#ROUGE score
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = []
for i in range(len(predictions)):
    scores.append(scorer.score(predictions[i], labels[i]))

rouge1 = [score['rouge1'].fmeasure for score in scores]
rouge2 = [score['rouge2'].fmeasure for score in scores]
rougeL = [score['rougeL'].fmeasure for score in scores]

np.mean(rouge1), np.mean(rouge2), np.mean(rougeL)



# %%
#METEOR score
from nltk.translate.meteor_score import meteor_score
meteor = []
for i in range(len(predictions)):
    meteor.append(meteor_score([labels[i]], predictions[i]))

np.mean(meteor)

# %%
#save the model
model.save_pretrained('model2')
tokenizer.save_pretrained('model2')


