import sentence_transformers
print(sentence_transformers.__version__)

from sentence_transformers import SentenceTransformer

# Load pre-trained sentence transformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Example sentences
sentences = ["This is a sentence.", "Another sentence for testing."]

# Get sentence embeddings
embeddings = model.encode(sentences)

# Print embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}\n")