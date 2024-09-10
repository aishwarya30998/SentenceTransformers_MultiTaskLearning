# SentenceTransformers_MultiTaskLearning
 Implementing Sentence transformers which takes in text and gives out fixed length embeddings and multiTask learning transformer
 # Step 1: Sentence Transformers
 Huggingface's transformers library provides a pre-trained model for this instrad of building a transformer from scratch which is time and resource consuming.
 I chose the distilbert-base-nli-stsb-mean-tokens model for efficiency and performance. It balances speed and accuracy by using a smaller transformer model.
 The output embedding is the mean of token embeddings, producing a fixed-size vector that can be used in downstream tasks.

 # Implementation steps
>> pip install transformers sentence-transformers torch
// running sentence_embedding file to get fixed length embeddings for given text.
>> python sentence_embedding.py 

# Step 2: Multi-Task Learning Expansion
... Task A: Sentence Classification - Adding a dense layer with softmax activation to classify sentences.
... Task B: NER or Sentiment Analysis - Adding another head for NER or sentiment analysis with a task-specific layer.

>> python multiTask.py 
# Output:
Classification logits: tensor([[ 0.6110, -0.2691,  0.1642],
                               [ 0.5485, -0.3351,  0.1157]], grad_fn=<AddmmBackward0>)
For the first sentence ([0.6110, -0.2691, 0.1642]), the highest logit is 0.6110, so the model would predict the sentence to belong to Class 0.
For the second sentence ([0.5485, -0.3351, 0.1157]), the highest logit is 0.5485, so the model would also predict this sentence to belong to Class 0.
Sentiment logits: tensor([[-0.2097, -0.7829],
                           [-0.3145, -0.6164]], grad_fn=<AddmmBackward0>)
For the first sentence ([-0.2097, -0.7829]), the highest logit is -0.2097, which would correspond to Class 0 (can assume as Positive sentiment).
For the second sentence ([-0.3145, -0.6164]), the highest logit is -0.3145, which would also predict Class 0 (Positive sentiment).

## Discussion Questions
1. Freezing Network Parts
... Freeze Transformer Backbone:
This would make sense if the transformer is pre-trained on large amounts of data and doesn't need fine-tuning for the task at hand.
It’s beneficial if you lack sufficient training data or want faster training.
... Freeze One Head:
This would be useful when one task is already well-trained, and you want to improve the performance of the other task without disrupting the trained task.

2. When to Use a Multi-Task Model vs. Separate Models
Multi-task model: Useful when both tasks share common features or representations (e.g., sentence structure) and when you want to reduce memory and training time.
Separate models: Makes sense when the tasks are completely different, sharing little to no information (e.g., text classification vs. image classification).

3. Handling Data Imbalance
Imbalanced Data: When Task A has abundant data, and Task B has limited data:
Use a higher loss weight for Task B to prioritize learning for that task.
Apply data augmentation or sampling strategies to balance the number of training examples.
Alternatively, pre-train Task A and fine-tune only on Task B.



