import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim)
        self.char_lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )

    def forward(self, char_input, char_lengths):
        embedded = self.char_emb(char_input)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            char_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, _ = self.char_lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        idx = (
            (char_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2))
        )
        last_outputs = outputs.gather(1, idx).squeeze(1)
        return last_outputs


class GCNLayer(nn.Module):
    def __init__(self, input_dim, gcn_dim):
        super().__init__()
        self.conv = GCNConv(input_dim, gcn_dim)

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))


class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz

        self.all1 = nn.Linear(self.hidden_sz + self.input_sz, self.hidden_sz)
        self.all2 = nn.Linear(
            self.hidden_sz + self.input_sz + self.g_sz, self.hidden_sz
        )
        self.all3 = nn.Linear(
            self.hidden_sz + self.input_sz + self.g_sz, self.hidden_sz
        )
        self.all4 = nn.Linear(self.hidden_sz + self.input_sz, self.hidden_sz)
        self.all11 = nn.Linear(self.hidden_sz + self.g_sz, self.hidden_sz)
        self.all44 = nn.Linear(self.hidden_sz + self.g_sz, self.hidden_sz)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)

        i = torch.sigmoid(self.all1(hx_concat))
        o = torch.sigmoid(self.all2(hxm_concat))
        f = torch.sigmoid(self.all3(hxm_concat))
        u = torch.tanh(self.all4(hx_concat))

        ii = torch.sigmoid(self.all11(hm_concat))
        uu = torch.tanh(self.all44(hm_concat))

        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x)
        return ht, Ct_x, Ct_m

    def forward(self, x, m):
        batch_sz, seq_sz, _ = x.size()
        ht = torch.zeros((batch_sz, self.hidden_sz), device=x.device)
        Ct_x = torch.zeros((batch_sz, self.hidden_sz), device=x.device)
        Ct_m = torch.zeros((batch_sz, self.hidden_sz), device=x.device)
        hidden_seq = []
        for t in range(seq_sz):
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m = self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2)
        return hidden_seq


class NNCRF(nn.Module):
    def __init__(self, config, use_char_embs=True, use_separate_lstms=True, use_syn_lstm=True):
        super().__init__()
        self.pad_idx = config.word2idx[config.PAD]
        self.device = config.device
        
        # Parámetros de configuración
        self.use_char_embs = use_char_embs
        self.use_separate_lstms = use_separate_lstms  
        self.use_syn_lstm = use_syn_lstm

        # BiLSTM para caracteres (solo si use_char_embs es True)
        if self.use_char_embs:
            self.char_bilstm = CharBiLSTM(
                config.char_vocab_size, config.char_emb_dim, config.char_hidden_dim
            )
            # Dimensión de características de palabras + caracteres
            word_feat_dim = config.word_emb_dim + config.char_hidden_dim
        else:
            word_feat_dim = config.word_emb_dim

        # Embedding para palabras
        self.word_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(config.word_embedding), freeze=False
        )

        # Embedding de dependencias (GCN)
        self.dep_embedding = nn.Embedding(len(config.dep2idx), config.dep_emb_dim)

        # GCN (conjunto de características combinadas)
        self.gcn = GCNLayer(
            word_feat_dim + config.dep_emb_dim, config.gcn_hidden_dim
        )

        # Configuración de LSTMs según parámetros
        if self.use_syn_lstm:
            # MyLSTM personalizada que usa información del grafo
            self.syn_lstm = MyLSTM(
                word_feat_dim + config.gcn_hidden_dim,
                config.hidden_dim,
                config.gcn_hidden_dim,
            )
        else:
            # LSTM estándar para NER (sin usar gcn_out)
            self.lstm_ner = nn.LSTM(
                word_feat_dim,
                config.hidden_dim // 2,
                num_layers=1,
                batch_first=True,
            )

        # LSTM separada para SA si use_separate_lstms es True
        if self.use_separate_lstms:
            self.lstm_sa = nn.LSTM(
                word_feat_dim,
                config.hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )

        # Capa de salida para NER
        self.hidden2tag_ner = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.label_size),
        )

        # Capa de salida para Sentimiento (SA)
        sa_input_dim = config.hidden_dim * 3  # Para 3 tipos de pooling (avg, max, last)
        self.hidden2tag_sa = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(sa_input_dim, 64),  # Reducir dimensionalidad
            nn.LayerNorm(64),  # Normalización para estabilidad
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
        )

    def focal_loss(self, logits, targets, alpha=0.5, gamma=2.0):
        """Focal Loss para combatir el desbalance de clases y ejemplos difíciles."""
        CE = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-CE)
        loss = alpha * (1 - pt) ** gamma * CE
        return loss.mean()

    def forward(
        self,
        word_inputs,
        char_inputs,
        char_lens,
        batch_graph,
        lengths,
        tags=None,
        sentiment_labels=None,
    ):
        batch_size, seq_len = word_inputs.size()

        # Paso 1: Obtener embeddings de palabras
        word_embeddings = self.word_emb(word_inputs)
        
        # Paso 2: Procesar características de caracteres si está habilitado
        if self.use_char_embs:
            char_features = self.char_bilstm(char_inputs, char_lens)
            char_features = char_features.view(batch_size, seq_len, -1)
            # Combinar word_embeddings con char_features
            word_features = torch.cat([word_embeddings, char_features], dim=-1)
        else:
            word_features = word_embeddings

        # Paso 3: Preparar entrada para GCN
        batch_graph.x = word_features.view(-1, word_features.shape[-1])

        # Paso 4: Propagar dependencias a través de GCN
        dep_embs = self.dep_embedding(batch_graph.dep_labels)
        edge_targets = batch_graph.edge_index[1]

        dep_embs_expanded = torch.zeros(
            batch_graph.x.size(0), self.dep_embedding.embedding_dim, device=self.device
        )
        dep_embs_expanded.index_add_(0, edge_targets, dep_embs)

        batch_graph.x = torch.cat([batch_graph.x, dep_embs_expanded], dim=-1)

        # Paso 5: GCN
        gcn_out = self.gcn(batch_graph.x, batch_graph.edge_index)
        gcn_out = gcn_out.view(batch_size, seq_len, -1)

        # Paso 6: Procesar secuencia para NER
        if self.use_syn_lstm:
            # Usando MyLSTM personalizada con información del grafo
            combined_input = torch.cat([word_features, gcn_out], dim=-1)
            lstm_out_ner = self.syn_lstm(combined_input, gcn_out)
        else:
            # Usando LSTM estándar sin información del grafo
            lstm_out_ner, _ = self.lstm_ner(word_features)

        # Paso 7: Procesar secuencia para SA
        if self.use_separate_lstms:
            # Usar una LSTM separada para SA
            lstm_out_sa, _ = self.lstm_sa(word_features)
        else:
            # Usar la misma salida de LSTM para SA
            lstm_out_sa = lstm_out_ner

        # Paso 8: Capa de salida para NER
        emissions_ner = self.hidden2tag_ner(lstm_out_ner)  # (B, S, label_size)
        mask = word_inputs != self.pad_idx

        # Paso 9: Capa de salida para Sentimiento (SA)
        # Combinar diferentes estrategias de pooling
        avg_pool = torch.mean(lstm_out_sa, dim=1)
        max_pool, _ = torch.max(lstm_out_sa, dim=1)
        last_token = lstm_out_sa[:, -1, :]
        pooled_features = torch.cat([avg_pool, max_pool, last_token], dim=1)

        emissions_sa = self.hidden2tag_sa(pooled_features)

        if tags is not None and sentiment_labels is not None:
            # Pérdida NER
            loss_ner = F.cross_entropy(
                emissions_ner.view(-1, emissions_ner.size(-1)),
                tags.view(-1),
                ignore_index=self.pad_idx,
            )

            # Pérdida SA
            loss_sa = self.focal_loss(
                emissions_sa,
                sentiment_labels,
                alpha=0.7,
                gamma=2.0,
            )

            # Pérdida combinada
            loss = loss_ner + loss_sa

            # Predicciones
            predicted_tags = emissions_ner.argmax(dim=-1)
            predicted_tags = [
                pred[:length].tolist() for pred, length in zip(predicted_tags, lengths)
            ]
            predicted_sentiments = emissions_sa.argmax(dim=1)
            return loss, predicted_tags, predicted_sentiments
        else:
            # Modo inferencia
            predicted_tags = emissions_ner.argmax(dim=-1)
            predicted_tags = [
                pred[:length].tolist() for pred, length in zip(predicted_tags, lengths)
            ]
            predicted_sentiments = emissions_sa.argmax(dim=1)
            return predicted_tags, predicted_sentiments
