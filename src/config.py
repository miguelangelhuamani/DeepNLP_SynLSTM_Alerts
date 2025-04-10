import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


class Config:
    def __init__(self, device):
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.O = "O"
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        self.device = device

        # Parámetros de embeddings
        self.word_emb_dim = 100  # tamaño de GloVe
        self.char_emb_dim = 30
        self.char_hidden_dim = 50
        self.hidden_dim = 200
        self.gcn_hidden_dim = 50
        self.dep_emb_dim = 30

        # Diccionarios
        self.word2idx = {}
        self.idx2word = []
        self.char2idx = {}
        self.idx2char = []
        self.label2idx = {}
        self.idx2label = []
        self.dep2idx = {}
        self.idx2dep = []

        # Embeddings de palabras (se cargará luego con GloVe)
        self.word_embedding = None

    def build_vocab(self, sentences, labels):
        # Vocabulario de palabras
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = [self.PAD, self.UNK]
        for sent in sentences:
            for word in sent:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)

        # Vocabulario de caracteres
        self.char2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2char = [self.PAD, self.UNK]
        for sent in sentences:
            for word in sent:
                for char in word:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx)
                        self.idx2char.append(char)

        # Vocabulario de etiquetas (con la corrección para 'O')
        self.label2idx = {self.PAD: 0, self.O: 1}
        self.idx2label = [self.PAD, self.O]
        for seq in labels:
            for tag in seq:
                if tag not in self.label2idx:
                    self.label2idx[tag] = len(self.label2idx)
                    self.idx2label.append(tag)

        # Vocab de dependencias
        self.build_dep_vocab(sentences)

        # Guardamos tamaños
        self.label_size = len(self.label2idx)
        self.char_vocab_size = len(self.char2idx)


    def build_dep_vocab(self, all_sentences):
        for sent in all_sentences:
            doc = nlp(" ".join(sent))
            for token in doc:
                if token.i != token.head.i:
                    dep_label = token.dep_
                    if dep_label not in self.dep2idx:
                        self.dep2idx[dep_label] = len(self.dep2idx)
                        self.idx2dep.append(dep_label)

    def init_embeddings(self, emb_path="../glove.6B.100d.txt"):
        """
        Carga embeddings preentrenados y los asigna a self.word_embedding.
        Si una palabra del vocabulario no está en GloVe, se inicializa aleatoriamente.
        """
        # Leemos GloVe o FastText
        pre_embeddings = {}
        with open(emb_path, "r", encoding="utf-8") as f:
            for line in f:
                split = line.strip().split()
                word = split[0]
                vector = np.array([float(val) for val in split[1:]])
                pre_embeddings[word] = vector

        # Inicializamos matriz de embeddings
        scale = np.sqrt(3.0 / self.word_emb_dim)
        vocab_size = len(self.word2idx)
        embedding_matrix = np.random.uniform(
            -scale, scale, (vocab_size, self.word_emb_dim)
        ).astype(np.float32)

        # Rellenamos la matriz con las palabras encontradas en GloVe
        found = 0
        for word, idx in self.word2idx.items():
            if word in pre_embeddings:
                embedding_matrix[idx] = pre_embeddings[word]
                found += 1

        print(f"Loaded {found}/{vocab_size} words from pre-trained embeddings.")
        self.word_embedding = embedding_matrix

