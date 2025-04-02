import os
import random
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import spacy
from transformers import pipeline
from sklearn.metrics import f1_score
from src.models import NNCRF

nlp = spacy.load("en_core_web_sm")
sa_pipeline = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment(text):
    result = sa_pipeline(text)
    label = result[0]['label']
    if label == 'LABEL_2':
        return 2  # POSITIVE
    elif label == 'LABEL_0':
        return 0  # NEGATIVE
    else:
        return 1  # NEUTRAL

def balance_by_sentiment(sentences, labels, sentiments, max_per_class=333):
    class_buckets = {0: [], 1: [], 2: []}
    
    for s, l, sa in zip(sentences, labels, sentiments):
        if len(class_buckets[sa]) < max_per_class:
            class_buckets[sa].append((s, l, sa))
    
    balanced = class_buckets[0] + class_buckets[1] + class_buckets[2]
    random.shuffle(balanced)

    balanced_sents = [b[0] for b in balanced]
    balanced_labels = [b[1] for b in balanced]
    balanced_sa = [b[2] for b in balanced]
    
    return balanced_sents, balanced_labels, balanced_sa


class NLPNERDataset(Dataset):
    def __init__(self, sentences, ner_labels, sentiment_labels, config):
        self.sentences = sentences
        self.ner_labels = ner_labels
        self.sentiment_labels = sentiment_labels
        self.config = config

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        ner_tags = self.ner_labels[idx]
        sentiment_tag = self.sentiment_labels[idx]  
        word_ids = [self.config.word2idx.get(w, self.config.word2idx[self.config.UNK]) for w in words]
        ner_tag_ids = [self.config.label2idx.get(t, self.config.label2idx[self.config.O]) for t in ner_tags]
        return words, word_ids, ner_tag_ids, sentiment_tag

def build_edge_index(sentence, config):
    doc = nlp(" ".join(sentence))
    edges, dep_labels = [], []
    for token in doc:
        if token.i != token.head.i:
            edges.append((token.head.i, token.i))
            dep_labels.append(config.dep2idx[token.dep_])
    if edges:
        edges = torch.tensor(edges, dtype=torch.long).t()
        dep_labels = torch.tensor(dep_labels, dtype=torch.long)
    else:
        edges = torch.zeros((2, 0), dtype=torch.long)
        dep_labels = torch.zeros((0,), dtype=torch.long)
    return edges, dep_labels

def nlp_collate_fn(batch, config):
    word_sequences = [torch.LongTensor(item[1]) for item in batch]
    label_sequences = [torch.LongTensor(item[2]) for item in batch]
    sentiment_labels = [item[3] for item in batch]
    sentences = [item[0] for item in batch]

    word_seq_lengths = torch.LongTensor([len(seq) for seq in word_sequences])
    max_seq_len = max(word_seq_lengths)

    char_sequences, char_lens = [], []
    for sent in sentences:
        padded_sent = sent + [config.PAD] * (max_seq_len - len(sent))
        for w in padded_sent:
            if w != config.PAD:
                chars = [config.char2idx.get(c, config.char2idx[config.UNK]) for c in w]
                char_sequences.append(torch.LongTensor(chars))
                char_lens.append(len(chars))
            else:
                char_sequences.append(torch.LongTensor([0]))
                char_lens.append(1)

    word_seq_tensor = pad_sequence(word_sequences, batch_first=True, padding_value=config.word2idx[config.PAD])
    label_seq_tensor = pad_sequence(label_sequences, batch_first=True, padding_value=config.label2idx[config.PAD])
    char_inputs = pad_sequence(char_sequences, batch_first=True, padding_value=0)
    char_lens_tensor = torch.LongTensor(char_lens)

    graphs = []
    for sent in sentences:
        edge_index, dep_labels = build_edge_index(sent, config)
        dummy_features = torch.zeros(len(sent), 1, device=config.device)
        g = Data(x=dummy_features, edge_index=edge_index.to(config.device), dep_labels=dep_labels.to(config.device))
        graphs.append(g)
    batch_graph = Batch.from_data_list(graphs)

    return (
        word_seq_tensor.to(config.device),
        word_seq_lengths.to(config.device),
        label_seq_tensor.to(config.device),
        char_inputs.to(config.device),
        char_lens_tensor.to(config.device),
        batch_graph,
        torch.tensor(sentiment_labels).to(config.device)
    )

def parse_umt_file(file_path):
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_tokens, current_labels = [], []
    for line in lines:
        line = line.strip()
        if line.startswith("IMGID:"):
            if current_tokens:
                sentences.append(current_tokens)
                labels.append(current_labels)
                current_tokens, current_labels = [], []
        elif line:
            if "\t" in line:
                token, tag = line.split("\t")
            else:
                token, tag = line.split(" ")
            current_tokens.append(token)
            current_labels.append(tag)
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)
    return sentences, labels

def load_umt_data():
    data_file = 'umt_data.pkl'
    if os.path.exists(data_file):
        print("Cargando datos UMT desde archivo.")
        with open(data_file, 'rb') as f:
            return pickle.load(f)

    train_sentences, train_labels = parse_umt_file("data/train.txt")
    val_sentences, val_labels = parse_umt_file("data/valid.txt")
    test_sentences, test_labels = parse_umt_file("data/test.txt")

    def generate_sentiment_labels(sentences):
        sa_labels = []
        for sent in sentences:
            text = " ".join(sent)
            sentiment = get_sentiment(text)
            sa_labels.append(sentiment)
        return sa_labels

    train_sa = generate_sentiment_labels(train_sentences)
    val_sa = generate_sentiment_labels(val_sentences)
    test_sa = generate_sentiment_labels(test_sentences)
    train_sentences, train_labels, train_sa = balance_by_sentiment(train_sentences, train_labels, train_sa, max_per_class=333)

    with open(data_file, 'wb') as f:
        pickle.dump((train_sentences, train_labels, train_sa, val_sentences, val_labels, val_sa, test_sentences, test_labels, test_sa), f)

    return train_sentences, train_labels, train_sa, val_sentences, val_labels, val_sa, test_sentences, test_labels, test_sa

def load_umt_loaders(config, batch_size=32, num_workers=0):
    train_sents, train_labels, train_sa, val_sents, val_labels, val_sa, test_sents, test_labels, test_sa = load_umt_data()
    config.build_vocab(train_sents + val_sents + test_sents, train_labels + val_labels + test_labels)
    config.init_embeddings()

    train_dataset = NLPNERDataset(train_sents, train_labels, train_sa, config)
    val_dataset = NLPNERDataset(val_sents, val_labels, val_sa, config)
    test_dataset = NLPNERDataset(test_sents, test_labels, test_sa, config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: nlp_collate_fn(x, config))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: nlp_collate_fn(x, config))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: nlp_collate_fn(x, config))

    return train_loader, val_loader, test_loader

class F1Score:
    def __init__(self):
        self.all_preds = []
        self.all_labels = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

    def compute(self) -> float:
        return f1_score(self.all_labels, self.all_preds, average='weighted')

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []

class Accuracy:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.correct += int(preds.eq(labels).sum().item())
        self.total += labels.shape[0]

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

def load_model_and_config(model_name: str) -> torch.nn.Module:
    model_path = f"models/{model_name}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    config = checkpoint['config']
    model = NNCRF(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def save_model(model: torch.nn.Module, config, name: str) -> None:
    if not os.path.isdir("models"):
        os.makedirs("models")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, f"models/{name}.pt")
