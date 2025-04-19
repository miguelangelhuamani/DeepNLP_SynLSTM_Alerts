import torch
from src.config import Config
from src.utils import load_model_and_config, nlp_collate_fn
from torch_geometric.data import Batch
import spacy
from src.alert_generation import AlertGenerator

# Load trained model and config
model_name = "combined_best_model"
model, config = load_model_and_config(model_name)
model.to(config.device)
model.eval()

# spaCy for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Example input text
texto = "German Chancellor Angela Merkel visited Paris to discuss climate change initiatives."
tokens = [token.text for token in nlp(texto)]

# Dummy labels for structure
fake_ner_tags = ["O"] * len(tokens)
fake_sa_label = 1  # positive by default

# Encode tokens and NER tags to indices
word_ids = [config.word2idx.get(w, config.word2idx[config.UNK]) for w in tokens]
ner_tag_ids = [config.label2idx.get(tag, config.label2idx[config.O]) for tag in fake_ner_tags]
batch = [(tokens, word_ids, ner_tag_ids, fake_sa_label)]
batch_inputs = nlp_collate_fn(batch, config)

# Unpack inputs
(
    word_seq_tensor,
    seq_lens,
    label_tensor,
    char_inputs,
    char_lens,
    batch_graph,
    sentiment_labels,
) = batch_inputs

# Inference
with torch.no_grad():
    predicted_tags, predicted_sentiment = model(
        word_seq_tensor,
        char_inputs,
        char_lens,
        batch_graph,
        seq_lens,
        tags=None,
        sentiment_labels=None,
    )

# Decode NER results
tags = [config.idx2label[tag] for tag in predicted_tags[0]]

# Convert sentiment prediction to string (only 0=negative, 1=positive considered)
sentiment_label = predicted_sentiment.item()
sentiment = "positive" if sentiment_label == 1 else "negative"

# Print results
print("Text:", texto)
print("Predicted Entities:")
for token, label in zip(tokens, tags):
    print(f"  {token}: {label}")
print("\nPredicted Sentiment:", sentiment)

# Generate alert
alert_gen = AlertGenerator()
alert = alert_gen.generate_multiple_alerts(texto, tokens, tags, sentiment)

print("\nGenerated Alert:", alert)
