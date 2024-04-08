from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from datasets import Dataset, load_metric
from datasets.features.features import ClassLabel
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=50,
    learning_rate=5e-5,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Read data from CSV
data = pd.read_csv('test_data.csv')

# Use train_test_split to create test and training sets
train, test = train_test_split(data, test_size=0.2)

# Create validation set
test, val = train_test_split(test, test_size=0.5)

# Trainer does not accept dataframes, will convert them to huggingface Datasets
train_hugging = Dataset.from_pandas(train)
test_hugging = Dataset.from_pandas(test)
val_hugging = Dataset.from_pandas(val)

# Need to add 'input_ids' which are a sequence of integers that represent the tokenized text
# Function to do this
def tokenize(sample):
    model_inps = tokenizer(sample["Sentence"], padding="max_length", truncation=True, max_length=128)
    return model_inps

# Tokenize inputs
train_tokenized = train_hugging.map(tokenize, batched=True)
test_tokenized = test_hugging.map(tokenize, batched=True)
val_tokenized = val_hugging.map(tokenize, batched=True)

# Put data in correct format for Trainer
train_tokenized = train_tokenized.remove_columns(['Sentence', '__index_level_0__'])
train_tokenized = train_tokenized.rename_column('Label', 'labels')
train_tokenized = train_tokenized.with_format('torch')
test_tokenized = test_tokenized.remove_columns(['Sentence', '__index_level_0__'])
test_tokenized = test_tokenized.rename_column('Label', 'labels')
test_tokenized = test_tokenized.with_format('torch')
val_tokenized = val_tokenized.remove_columns(['Sentence', '__index_level_0__'])
val_tokenized = val_tokenized.rename_column('Label', 'labels')
val_tokenized = val_tokenized.with_format('torch')

# Ensure that the labels are of ClassLabel
train_tokenized = train_tokenized.cast_column('labels', ClassLabel(num_classes=2, names=['not axiom', 'axiom']))
test_tokenized = test_tokenized.cast_column('labels', ClassLabel(num_classes=2, names=['not axiom', 'axiom']))
val_tokenized = val_tokenized.cast_column('labels', ClassLabel(num_classes=2, names=['not axiom', 'axiom']))

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train the model
trainer.train()

# Get predictions on test set
pred = trainer.predict(test_tokenized)

# Print the first x predictions
x = 10
for i in range(x):
    sentence = test_hugging[i]["Sentence"]
    pred_label = pred.label_ids[i]
    actual_label = test_hugging[i]["Label"]
    print(f"Sentence: {sentence}, Actual Label: {actual_label}, Predicted Label: {pred_label}")