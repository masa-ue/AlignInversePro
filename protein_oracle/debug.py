import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# # Load your data
# data = pd.read_csv('your_protein_data.csv')
# sequences = data['sequence'].tolist()
# properties = data['property'].tolist()

# # Split the data
# train_seq, val_seq, train_prop, val_prop = train_test_split(sequences, properties, test_size=0.2, random_state=42)

# Define a custom dataset
class ProteinDataset(Dataset):
    def __init__(self, sequences, properties, tokenizer, max_length=512):
        self.sequences = sequences
        self.properties = properties
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        prop = self.properties[idx]

        encoding = self.tokenizer(seq, 
                                  truncation=True, 
                                  padding='max_length', 
                                  max_length=self.max_length, 
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(prop, dtype=torch.float)
        }

# Load the ESM2 tokenizer and model
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Create datasets
# train_dataset = ProteinDataset(train_seq, train_prop, tokenizer)
# val_dataset = ProteinDataset(val_seq, val_prop, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("./fine_tuned_esm2")

# Optionally, evaluate the model
# eval_results = trainer.evaluate()
# print(eval_results)

# Example of using the fine-tuned model for prediction
# def predict_property(sequence):
#     inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     return outputs.logits.item()

# # Example usage
# test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# predicted_property = predict_property(test_sequence)
# print(f"Predicted property for the test sequence: {predicted_property}")


# Example of using the fine-tuned model for prediction with batches
def predict_properties(sequences):
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.logits.squeeze()

# Example usage
test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "SATVSEINSETDFVAKNDQFIALTKDTTAHIQSNSLQSVEELHSSTINGVKFEEYLKSQIATIGENLVVRRFATLKAGANGVVNGYIHTNGRVGVVIAAACDSAEVASKSRDLLRQICMH"
]
predicted_properties = predict_properties(test_sequences)
for seq, prop in zip(test_sequences, predicted_properties):
    print(f"Predicted property for sequence: {seq[:10]}... is {prop.item()}")