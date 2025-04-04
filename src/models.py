import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchcrf import CRF


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
    def __init__(self, config):
        super().__init__()
        self.pad_idx = config.word2idx[config.PAD]
        self.device = config.device

        # BiLSTM para caracteres
        self.char_bilstm = CharBiLSTM(
            config.char_vocab_size, config.char_emb_dim, config.char_hidden_dim
        )

        # Embedding para palabras
        self.word_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(config.word_embedding), freeze=False
        )
        
        # Embedding de dependencias (GCN)
        self.dep_embedding = nn.Embedding(len(config.dep2idx), config.dep_emb_dim)

        # GCN (conjunto de características combinadas)
        self.gcn = GCNLayer(
            config.word_emb_dim + config.char_hidden_dim + config.dep_emb_dim, 
            config.gcn_hidden_dim
        )

        # Syn-LSTM para combinar todas las características
        self.syn_lstm = MyLSTM(
            config.word_emb_dim + config.char_hidden_dim + config.gcn_hidden_dim,
            config.hidden_dim,
            config.gcn_hidden_dim,
        )

        # Capa de salida para NER
        self.hidden2tag_ner = nn.Linear(config.hidden_dim, config.label_size)
        self.crf = CRF(config.label_size, batch_first=True)

        # Capa de salida para Sentimiento (SA)
        self.hidden2tag_sa = nn.Linear(config.hidden_dim, 3)  # Tres clases para el sentimiento (positivo, negativo, neutral)

    def forward(
        self, word_inputs, char_inputs, char_lens, batch_graph, lengths, tags=None, sentiment_labels=None
    ):
        batch_size, seq_len = word_inputs.size()

        # Paso 1: Obtener características a partir de BiLSTM de caracteres
        char_features = self.char_bilstm(char_inputs, char_lens)
        char_features = char_features.view(batch_size, seq_len, -1)  # Recuperamos (B, S, dim)

        # Paso 2: Obtener embeddings de palabras
        word_embeddings = self.word_emb(word_inputs)

        # Paso 3: Concatenar word + char features
        input_gcn = torch.cat([word_embeddings, char_features], dim=-1)
        batch_graph.x = input_gcn.view(-1, input_gcn.shape[-1])  # (batch*seq_len, dim)

        # Paso 4: Propagar dependencias a través de GCN
        dep_embs = self.dep_embedding(batch_graph.dep_labels)  # (num_edges, dep_emb_dim)
        edge_targets = batch_graph.edge_index[1]

        dep_embs_expanded = torch.zeros(
            batch_graph.x.size(0), self.dep_embedding.embedding_dim, device=self.device
        )
        dep_embs_expanded.index_add_(0, edge_targets, dep_embs)

        batch_graph.x = torch.cat([batch_graph.x, dep_embs_expanded], dim=-1)

        # Paso 5: GCN
        gcn_out = self.gcn(batch_graph.x, batch_graph.edge_index)
        gcn_out = gcn_out.view(batch_size, seq_len, -1)

        # Paso 6: Combinar todas las características
        combined_input = torch.cat([word_embeddings, char_features, gcn_out], dim=-1)
        lstm_out = self.syn_lstm(combined_input, gcn_out)

        # Paso 7: Capa de salida para NER
        emissions_ner = self.hidden2tag_ner(lstm_out)
        mask = word_inputs != self.pad_idx

        # Paso 8: Capa de salida para Sentimiento (SA)
        emissions_sa = self.hidden2tag_sa(lstm_out)
        emissions_sa_reduced = emissions_sa.mean(dim=1)
        #print(f"emissions_sa shape: {emissions_sa_reduced.shape}")
        #print(f"sentiment_labels shape: {sentiment_labels.shape}")


        if tags is not None and sentiment_labels is not None:
            loss_ner = -self.crf(emissions_ner, tags, mask=mask)
            # Definir pesos: inversamente proporcionales a la frecuencia
            class_weights = torch.tensor([3.0, 1.0, 2.0], device=self.device)  # Neg: 10%, Neu: 60%, Pos: 30%
            loss_sa = F.cross_entropy(emissions_sa_reduced, sentiment_labels, weight=class_weights)

            loss = loss_ner + loss_sa

            decoded_tags = self.crf.decode(emissions_ner, mask=mask)
            predicted_sentiments = emissions_sa_reduced.argmax(dim=1)
            #print(f"emissions: {emissions_sa_reduced}")
            return loss, decoded_tags, predicted_sentiments
        else:
            # En evaluación
            decoded_tags = self.crf.decode(emissions_ner, mask=mask)
            predicted_sentiments = emissions_sa_reduced.argmax(dim=1)
            return decoded_tags, predicted_sentiments

