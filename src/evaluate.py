import os
import torch
from torch.utils.data import DataLoader
from src.utils import load_nlp_data, Accuracy, load_model_and_config, set_seed, F1Score
from src.config import Config

# Configurar el dispositivo
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


# Configurar semillas y número de hilos
set_seed(42)
torch.set_num_threads(8)

# Variables estáticas
MODEL_DIR: str = "models"

def evaluate(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, config) -> float:
    """
    Evalúa el modelo en el conjunto de test.

    Args:
        model: Modelo de Torch.
        test_loader: DataLoader con los datos de prueba.
        device: Dispositivo de ejecución (CPU/GPU).
        config: Configuración para obtener PAD idx y otras infos.

    Returns:
        Precisión del modelo en test.
    """
    model.eval()
    accuracy = Accuracy()
    f1 = F1Score()
    test_loss = 0.0
    
    with torch.no_grad():
        for (
            word_seq_tensor,
            seq_lens,
            label_tensor,
            char_inputs,
            char_lens,
            batch_graph,
        ) in test_loader:
            word_seq_tensor = word_seq_tensor.to(device)
            seq_lens = seq_lens.to(device)
            label_tensor = label_tensor.to(device)
            char_inputs = char_inputs.to(device)
            char_lens = char_lens.to(device)
            batch_graph = batch_graph.to(device)

            loss_value, predictions = model(
                word_seq_tensor, char_inputs, char_lens, batch_graph, seq_lens, tags=label_tensor
            )

            test_loss += loss_value.item()
            flat_preds = torch.tensor(
                [p for seq in predictions for p in seq], device=device
            )
            flat_labels = label_tensor.view(-1)
            mask = flat_labels != config.label2idx[config.PAD]
            accuracy.update(flat_preds, flat_labels[mask])
            f1.update(flat_preds, flat_labels[mask])
            mostrar = False
            if mostrar:
                for i in range(min(10, len(word_seq_tensor))):  
                    word_seq = word_seq_tensor[i].cpu().numpy()
                    true_labels = label_tensor[i].cpu().numpy()
                    pred_labels = predictions[i]  

                    
                    words = [config.idx2word[idx] for idx in word_seq if idx != config.word2idx[config.PAD]]


                    true_label_list = [config.idx2label[true_labels[j]] if true_labels[j] != config.label2idx[config.PAD] else 'PAD' for j in range(len(true_labels))]
                    pred_label_list = [config.idx2label[pred_labels[j]] if pred_labels[j] != config.label2idx[config.PAD] else 'PAD' for j in range(len(pred_labels))]
                    print(f"Words: {' '.join(words)}")
                    print(f"True labels: {' '.join(true_label_list)}")
                    print(f"Predicted labels: {' '.join(pred_label_list)}")
                print("-" * 30)

    test_loss /= len(test_loader)
    test_acc = accuracy.compute()
    test_f1 = f1.compute()

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    return test_acc


def main():
    model_name = "ner_model_synlstm2"
    
    model, config = load_model_and_config(model_name)

    model.to(device)

    _, _, test_loader = load_nlp_data(config, batch_size=100)

    evaluate(model, test_loader, device, config)

if __name__ == "__main__":
    main()