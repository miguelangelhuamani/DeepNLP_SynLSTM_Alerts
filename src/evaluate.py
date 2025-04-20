import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from src.utils import (
    load_umt_loaders,
    Accuracy,
    load_model_and_config,
    set_seed,
    F1Score,
)
from src.config import Config
import argparse

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


def generate_comparison_plots(all_results, model_names):
    """
    Genera gráficos comparativos para todos los modelos.
    
    Args:
        all_results: Lista de diccionarios con métricas para cada modelo
        model_names: Lista con los nombres de los modelos
    """
    # Crear directorio para guardar gráficos
    os.makedirs("plots", exist_ok=True)
    
    # Métricas a comparar
    metrics = [
        ("test_acc_ner", "NER Accuracy"),
        ("test_f1_ner", "NER F1 Score"),
        ("test_acc_sa", "Sentiment Accuracy"),
        ("test_f1_sa", "Sentiment F1 Score")
    ]
    
    # Generar un gráfico para cada métrica
    for metric_key, metric_title in metrics:
        plt.figure(figsize=(12, 6))
        
        # Datos para el gráfico
        values = [result[metric_key] for result in all_results]
        x = np.arange(len(model_names))
        width = 0.6
        
        # Crear barras
        bars = plt.bar(x, values, width, color='skyblue')
        
        # Añadir valor numérico sobre cada barra
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Configurar etiquetas y título
        plt.xlabel('Modelo')
        plt.ylabel(metric_title)
        plt.title(f'Comparativa de {metric_title} entre modelos')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylim(0, 1.05)  # Ajustar los límites del eje y
        plt.tight_layout()
        
        # Guardar gráfico
        plt.savefig(f"plots/{metric_key}_comparison.png")
        plt.close()
    
    # Crear gráfico combinado para test
    plt.figure(figsize=(14, 10))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    # Crear barras para cada métrica
    plt.bar(x - width*1.5, [r["test_acc_ner"] for r in all_results], width, label='NER Accuracy', color='skyblue')
    plt.bar(x - width/2, [r["test_f1_ner"] for r in all_results], width, label='NER F1', color='royalblue')
    plt.bar(x + width/2, [r["test_acc_sa"] for r in all_results], width, label='SA Accuracy', color='lightcoral')
    plt.bar(x + width*1.5, [r["test_f1_sa"] for r in all_results], width, label='SA F1', color='firebrick')
    
    # Configurar etiquetas
    plt.xlabel('Modelo')
    plt.ylabel('Puntuación')
    plt.title('Comparativa de todas las métricas entre modelos')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar gráfico combinado
    plt.savefig("plots/all_metrics_comparison.png")
    plt.close()
    
    print(f"\nGráficos guardados en el directorio 'plots/'")


def main(model_names=None):
    """
    Evalúa varios modelos y genera comparativas gráficas.
    
    Args:
        model_names: Lista de nombres de modelos a evaluar
    """
    if model_names is None:
        model_names = ["combined_best_model"]
    
    # Cargar una vez los loaders
    config_tmp = load_model_and_config(model_names[0])[1]
    train_loader, val_loader, test_loader = load_umt_loaders(config_tmp, batch_size=32)
    
    all_results = []
    
    for model_name in model_names:
        print("\n" + "=" * 50)
        print(f"EVALUACIÓN DEL MODELO: {model_name}")
        print("=" * 50)
        
        model, config = load_model_and_config(model_name)
        model.to(device)
        
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
        
        # Combinar todas las métricas en un solo diccionario
        combined_metrics = {}
        combined_metrics.update(train_metrics)
        combined_metrics.update(val_metrics)
        combined_metrics.update(test_metrics)
        
        all_results.append(combined_metrics)
    
    # Generar gráficos comparativos si hay más de un modelo
    if len(model_names) > 1:
        print("\n" + "=" * 50)
        print("GENERANDO GRÁFICOS COMPARATIVOS")
        print("=" * 50)
        generate_comparison_plots(all_results, model_names)


if __name__ == "__main__": 
    main(['combined_best_model'])
