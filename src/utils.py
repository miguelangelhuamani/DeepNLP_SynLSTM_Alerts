import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import spacy
from src.models import NNCRF  # Asegúrate de usar el modelo correcto

from datasets import load_dataset
nlp = spacy.load("en_core_web_sm")
from transformers import pipeline

sa_pipeline = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment(text):
    result = sa_pipeline(text)
    return result[0]['label']  

class NLPNERDataset(Dataset):
    def __init__(self, sentences, ner_labels, sentiment_labels, config):
        self.sentences = sentences
        self.ner_labels = ner_labels
        self.sentiment_labels = sentiment_labels  # Añadir etiquetas de sentimiento
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
    sentiment_labels = [item[3] for item in batch]  # Obtener las etiquetas de sentimiento
    sentences = [item[0] for item in batch]

    word_seq_lengths = torch.LongTensor([len(seq) for seq in word_sequences])
    max_seq_len = max(word_seq_lengths)

    char_sequences = []
    char_lens = []

    for sent in sentences:
        padded_sent = sent + [config.PAD] * (max_seq_len - len(sent))  # rellenamos hasta max_seq_len
        for w in padded_sent:
            if w != config.PAD:
                chars = [config.char2idx.get(c, config.char2idx[config.UNK]) for c in w]
                char_sequences.append(torch.LongTensor(chars))
                char_lens.append(len(chars))
            else:
                # Dummy input para el token PAD
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
        g = Data(
            x=dummy_features,
            edge_index=edge_index.to(config.device),
            dep_labels=dep_labels.to(config.device),
        )
        graphs.append(g)
    batch_graph = Batch.from_data_list(graphs)

    return (
        word_seq_tensor.to(config.device),
        word_seq_lengths.to(config.device),
        label_seq_tensor.to(config.device),
        char_inputs.to(config.device),
        char_lens_tensor.to(config.device),
        batch_graph,
        torch.tensor(sentiment_labels).to(config.device)  # Pasar también las etiquetas de sentimiento
    )




from datasets import load_dataset

import os
import pickle
from datasets import load_dataset

def load_conll2003():
    # Nombre del archivo donde se guardarán los datos procesados
    data_file = 'conll2003_data.pkl'

    # Si el archivo ya existe, cargar los datos desde allí
    if os.path.exists(data_file):
        print("Cargando datos desde el archivo.")
        with open(data_file, 'rb') as f:
            return pickle.load(f)
    
    # Si no existe el archivo, cargar y procesar los datos
    dataset = load_dataset("conll2003")
    
    # Crear listas vacías para los datos
    train_sents, train_labels, train_sa_labels = [], [], []
    val_sents, val_labels, val_sa_labels = [], [], []
    test_sents, test_labels, test_sa_labels = [], [], []

    # Función para obtener las etiquetas de sentimiento
    def generate_sentiment_labels(sentences):
        sentiment_labels = []
        for sentence in sentences:
            text = ' '.join(sentence)
            sentiment = get_sentiment(text)
            if sentiment == 'POSITIVE':
                sentiment_labels.append(2)  # Etiqueta para positivo
            elif sentiment == 'NEGATIVE':
                sentiment_labels.append(0)  # Etiqueta para negativo
            else:
                sentiment_labels.append(1)  # Etiqueta para neutral
        return sentiment_labels

    # Separar los datos
    for ex in dataset['train']:
        tokens = ex['tokens']
        labels = ex['ner_tags']
        train_sents.append(tokens)
        train_labels.append(labels)
    
    for ex in dataset['validation']:
        tokens = ex['tokens']
        labels = ex['ner_tags']
        val_sents.append(tokens)
        val_labels.append(labels)

    for ex in dataset['test']:
        tokens = ex['tokens']
        labels = ex['ner_tags']
        test_sents.append(tokens)
        test_labels.append(labels)
    
    # Obtener las etiquetas de sentimiento para cada conjunto
    train_sa_labels = generate_sentiment_labels(train_sents)
    val_sa_labels = generate_sentiment_labels(val_sents)
    test_sa_labels = generate_sentiment_labels(test_sents)

    # Mapeo de números a etiquetas reales:
    label_list = dataset['train'].features['ner_tags'].feature.names
    train_labels = [[label_list[l] for l in seq] for seq in train_labels]
    val_labels = [[label_list[l] for l in seq] for seq in val_labels]
    test_labels = [[label_list[l] for l in seq] for seq in test_labels]

    # Guardar los datos procesados en un archivo para usarlos posteriormente
    with open(data_file, 'wb') as f:
        pickle.dump((train_sents, train_labels, train_sa_labels, val_sents, val_labels, val_sa_labels, test_sents, test_labels, test_sa_labels), f)
    
    return train_sents, train_labels, train_sa_labels, val_sents, val_labels, val_sa_labels, test_sents, test_labels, test_sa_labels


def load_nlp_data(config, batch_size=32, num_workers=0):
    # Cargar datos y etiquetas de NER y SA
    train_sents, train_labels, train_sa_labels, val_sents, val_labels, val_sa_labels, test_sents, test_labels, test_sa_labels = load_conll2003()

    # Construir vocabulario y embeddings
    config.build_vocab(
        train_sents + val_sents + test_sents, train_labels + val_labels + test_labels
    )
    config.init_embeddings()

    # Crear Dataset + DataLoader
    train_dataset = NLPNERDataset(train_sents, train_labels, train_sa_labels, config)
    val_dataset = NLPNERDataset(val_sents, val_labels, val_sa_labels, config)
    test_dataset = NLPNERDataset(test_sents, test_labels, test_sa_labels, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: nlp_collate_fn(x, config),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: nlp_collate_fn(x, config),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: nlp_collate_fn(x, config),
    )

    return train_loader, val_loader, test_loader


from sklearn.metrics import f1_score

class F1Score:
    def __init__(self):
        self.all_preds = []
        self.all_labels = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        # Convierte las predicciones y las etiquetas a numpy para el cálculo de F1
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        # Añadir las predicciones y etiquetas al acumulador
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)

    def compute(self) -> float:
        # Calcula el F1-score con el promedio ponderado
        return f1_score(self.all_labels, self.all_preds, average='weighted')

    def reset(self) -> None:
        # Resetea los acumuladores para la siguiente evaluación
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

# Función para cargar el modelo
def load_model_and_config(model_name: str) -> torch.nn.Module:
    model_path = f"models/{model_name}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    
    # Cargar el checkpoint con weights_only=False
    checkpoint = torch.load(model_path, weights_only=False)  
    config = checkpoint['config']
    model = NNCRF(config)  
    model.load_state_dict(checkpoint['model_state_dict'])  
      
    
    return model, config



# Función para configurar la semilla para reproducibilidad
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



