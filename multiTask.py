import torch
import torch.nn as nn  # Ensure this is imported
from transformers import BertModel, BertTokenizer

# Define the model (your existing MultiTaskModel code)
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Task A: Sentence Classification (3 classes as an example)
        self.classification_head = nn.Linear(self.bert.config.hidden_size, 3)
        
        # Task B: Sentiment Analysis (2 classes: Positive, Negative)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled output of [CLS] token
        
        # Task A: Sentence Classification
        classification_logits = self.classification_head(pooled_output)
        
        # Task B: Sentiment Analysis
        sentiment_logits = self.sentiment_head(pooled_output)
        
        return classification_logits, sentiment_logits

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MultiTaskModel()

# Example input sentences
sentences = ["This is a test sentence.", "I love programming."]

# Tokenize input sentences
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Run the model
classification_logits, sentiment_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Print the logits for Task A (classification) and Task B (sentiment analysis)
print("Classification logits:", classification_logits)
print("Sentiment logits:", sentiment_logits)