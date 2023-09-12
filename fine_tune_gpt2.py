#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 03:45:34 2023

@author: trootech
"""

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load your dataset from CSV
data = pd.read_csv(r"car_data.csv")

# Preprocess your data
# For this example, we'll concatenate all columns into a single string for each row.
data['text'] = data.apply(lambda row: '\t'.join(map(str, row)), axis=1)

# Save the preprocessed data to a text file
data['text'].to_csv("preprocessed_data.txt", header=None, index=None, sep="\n")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # or a specific GPT-2 variant if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare your dataset and tokenize it
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="preprocessed_data.txt",  # Replace with the path to your preprocessed dataset
    block_size=128  # Adjust block size as needed
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Adjust as needed
    per_device_train_batch_size=64,  # Adjust batch size
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine-tuned-gpt2")
tokenizer.save_pretrained("fine-tuned-gpt2")

# Inference: You can now use the fine-tuned model for generating text based on tabular data.
########################################################################################
from transformers import pipeline, set_seed

# Load a fine-tuned GPT-2 or GPT-3 model here
# Example using GPT-2:
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
model = GPT2LMHeadModel.from_pretrained("fine-tuned-gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("fine-tuned-gpt2")

# Define the questions
questions = [
    "What is the average mpg (Miled Per Gallon) of used cars in the dataset?",
    "What is the most popular make and model of used car in the dataset?",
    "What is the correlation between mpg and price?",
    "What is the oldest used car in the dataset?",
    "What is the most popular color of used car in the dataset?",
    "What is the price of the oldest Tesla in the dataset?",
]

# Initialize the text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Set a random seed for reproducibility
set_seed(9)

# Store the answers
answers = []

# Loop through the questions and generate answers
for question in questions:
    answer = generator(question, max_length=80, num_return_sequences=1)
    answers.append(answer[0]["generated_text"])

# Display the answers
for i, answer in enumerate(answers):
    #print(f"Q{i+1}: {questions[i]}\nA{i+1}: {answer}\n")
    print(f"Q{i+1}: {answer}\n")


























