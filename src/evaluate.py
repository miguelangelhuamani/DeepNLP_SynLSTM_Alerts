import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import (
    load_umt_loaders,
    Accuracy,
    load_model_and_config,
    set_seed,
    F1Score,
)
from src.config import Config

# Configurar el dispositivo
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Configurar semillas y número de hilos
set_seed(42)
torch.set_num_threads(8)

# Variables estáticas
MODEL_DIR: str = "models"


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config,
    dataset_name: str = "Test",
) -> tuple:
    """
    Evalúa el modelo en cualquier conjunto de datos, calculando métricas separadas para NER y SA.

    Args:
        model: Modelo de Torch.
        data_loader: DataLoader con los datos a evaluar.
        device: Dispositivo de ejecución (CPU/GPU).
        config: Configuración para obtener PAD idx y otras infos.
        dataset_name: Nombre del conjunto para mostrar en los resultados.

    Returns:
        Tuple con diccionario de métricas y pérdida.
    """
    model.eval()
    total_loss = 0.0

    # Métricas para NER
    accuracy_ner = Accuracy()
    f1_ner = F1Score()

    # Métricas para SA
    accuracy_sa = Accuracy()
    f1_sa = F1Score()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            # Desempaquetar el batch
            (
                word_seq_tensor,
                seq_lens,
                label_tensor,
                char_inputs,
                char_lens,
                batch_graph,
                sentiment_labels,
            ) = batch

            # Mover tensores al dispositivo
            word_seq_tensor = word_seq_tensor.to(device)
            seq_lens = seq_lens.to(device)
            label_tensor = label_tensor.to(device)
            char_inputs = char_inputs.to(device)
            char_lens = char_lens.to(device)
            batch_graph = batch_graph.to(device)
            sentiment_labels = sentiment_labels.to(device)

            # Inferencia
            loss_value, preds_ner, preds_sa = model(
                word_seq_tensor,
                char_inputs,
                char_lens,
                batch_graph,
                seq_lens,
                tags=label_tensor,
                sentiment_labels=sentiment_labels,
            )

            total_loss += loss_value.item()

            # Cálculo de métricas para NER
            flat_preds_ner = []
            for pred_seq, length in zip(preds_ner, seq_lens):
                if isinstance(pred_seq, torch.Tensor):
                    flat_preds_ner.extend(pred_seq[:length].tolist())
                else:
                    flat_preds_ner.extend(pred_seq[:length])
            flat_preds_ner = torch.tensor(flat_preds_ner, device=device)

            flat_labels_ner = label_tensor.view(-1)
            mask_ner = flat_labels_ner != config.label2idx[config.PAD]
            accuracy_ner.update(flat_preds_ner, flat_labels_ner[mask_ner])
            f1_ner.update(flat_preds_ner, flat_labels_ner[mask_ner])

            # Cálculo de métricas para SA
            accuracy_sa.update(preds_sa, sentiment_labels)
            f1_sa.update(preds_sa, sentiment_labels)

    # Calcular métricas finales
    loss = total_loss / len(data_loader)
    acc_ner = accuracy_ner.compute()
    f1_score_ner = f1_ner.compute()
    acc_sa = accuracy_sa.compute()
    f1_score_sa = f1_sa.compute()

    # Mostrar resultados
    print(f"\n{dataset_name} Results:")
    print(f"  Loss:       {loss:.4f}")
    print(f"  NER:        Accuracy: {acc_ner:.4f}, F1: {f1_score_ner:.4f}")
    print(f"  Sentiment:  Accuracy: {acc_sa:.4f}, F1: {f1_score_sa:.4f}")

    metrics = {
        f"{dataset_name.lower()}_loss": loss,
        f"{dataset_name.lower()}_acc_ner": acc_ner,
        f"{dataset_name.lower()}_f1_ner": f1_score_ner,
        f"{dataset_name.lower()}_acc_sa": acc_sa,
        f"{dataset_name.lower()}_f1_sa": f1_score_sa,
    }

    return metrics


def main():
    model_name = "combined_best_model"
    model, config = load_model_and_config(model_name)
    model.to(device)

    # Cargar los loaders
    train_loader, val_loader, test_loader = load_umt_loaders(config, batch_size=32)

    print("\n" + "=" * 50)
    print("EVALUACIÓN COMPLETA DEL MODELO")
    print("=" * 50)

    # Evaluar en test
    test_metrics = evaluate(model, test_loader, device, config, "Test")

    # Evaluar en validación
    val_metrics = evaluate(model, val_loader, device, config, "Validation")

    # Evaluar en entrenamiento
    train_metrics = evaluate(model, train_loader, device, config, "Train")

    # Mostrar resumen comparativo
    print("\n" + "=" * 50)
    print("RESUMEN COMPARATIVO")
    print("=" * 50)
    print(f"                  Train       Validation  Test")
    print(
        f"NER Accuracy:     {train_metrics['train_acc_ner']:.4f}       {val_metrics['validation_acc_ner']:.4f}      {test_metrics['test_acc_ner']:.4f}"
    )
    print(
        f"NER F1:           {train_metrics['train_f1_ner']:.4f}       {val_metrics['validation_f1_ner']:.4f}      {test_metrics['test_f1_ner']:.4f}"
    )
    print(
        f"Sentiment Acc:    {train_metrics['train_acc_sa']:.4f}       {val_metrics['validation_acc_sa']:.4f}      {test_metrics['test_acc_sa']:.4f}"
    )
    print(
        f"Sentiment F1:     {train_metrics['train_f1_sa']:.4f}       {val_metrics['validation_f1_sa']:.4f}      {test_metrics['test_f1_sa']:.4f}"
    )


if __name__ == "__main__":
    main()
