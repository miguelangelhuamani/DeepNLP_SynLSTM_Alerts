import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from src.models import NNCRF
from src.utils import load_umt_loaders, Accuracy, save_model, set_seed, F1Score

from src.config import Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(device)

# Configuración global
set_seed(42)
torch.set_num_threads(8)

EPOCHS = 7
BATCH_SIZE = 100
LEARNING_RATE = 0.2
DECAY = 0.1
L2_REG = 1e-8


def adjust_lr(optimizer, epoch):
    lr = LEARNING_RATE / (1 + DECAY * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"[Epoch {epoch}] Learning rate adjusted to {lr:.6f}")



def train(model, optimizer, train_loader, device, epoch, config):
    model.train()
    train_loss = 0.0

    # Métricas para NER
    accuracy_ner = Accuracy()
    f1_ner = F1Score()

    # Métricas para SA
    accuracy_sa = Accuracy()
    f1_sa = F1Score()

    for (
        word_seq_tensor,
        seq_lens,
        label_tensor,
        char_inputs,
        char_lens,
        edge_index,
        sentiment_labels
    ) in train_loader:
        optimizer.zero_grad()

        # El modelo ahora devuelve tanto las predicciones de NER como de SA
        loss, preds_ner, preds_sa = model(
            word_seq_tensor,
            char_inputs,
            char_lens,
            edge_index,
            seq_lens,
            tags=label_tensor,
            sentiment_labels=sentiment_labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        # Cálculo de las métricas para NER
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

        # Cálculo de las métricas para SA
        #print(preds_sa)
        flat_preds_sa = preds_sa
        accuracy_sa.update(flat_preds_sa, sentiment_labels)
        from collections import Counter
        #print(Counter(sentiment_labels.tolist()))

        f1_sa.update(flat_preds_sa, sentiment_labels)

    # Cálculo de las métricas por epoch para NER y SA
    train_acc_ner = accuracy_ner.compute()
    train_f1_ner = f1_ner.compute()

    train_acc_sa = accuracy_sa.compute()
    train_f1_sa = f1_sa.compute()

    return train_loss / len(train_loader), {
        'train_acc_ner': train_acc_ner,
        'train_f1_ner': train_f1_ner,
        'train_acc_sa': train_acc_sa,
        'train_f1_sa': train_f1_sa,
    }



def validate(model, val_loader, device, config):
    model.eval()
    val_loss = 0.0

    # Métricas para NER
    accuracy_ner = Accuracy()
    f1_ner = F1Score()

    # Métricas para SA
    accuracy_sa = Accuracy()
    f1_sa = F1Score()

    with torch.no_grad():
        for (
            word_seq_tensor,
            seq_lens,
            label_tensor,
            char_inputs,
            char_lens,
            edge_index,
            sentiment_labels
        ) in val_loader:
            
            loss_value, preds_ner, preds_sa = model(
                word_seq_tensor,
                char_inputs,
                char_lens,
                edge_index,
                seq_lens,
                tags=label_tensor,
                sentiment_labels=sentiment_labels
            )

            val_loss += loss_value.item()

            # Cálculo de las métricas para NER
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

            # Cálculo de las métricas para SA
            flat_preds_sa = preds_sa
            accuracy_sa.update(flat_preds_sa, sentiment_labels)
            f1_sa.update(flat_preds_sa, sentiment_labels)

    # Cálculo de las métricas por epoch para NER y SA
    val_acc_ner = accuracy_ner.compute()
    val_f1_ner = f1_ner.compute()

    val_acc_sa = accuracy_sa.compute()
    val_f1_sa = f1_sa.compute()

    return val_loss / len(val_loader), {
        'val_acc_ner': val_acc_ner,
        'val_f1_ner': val_f1_ner,
        'val_acc_sa': val_acc_sa,
        'val_f1_sa': val_f1_sa,
    }



def main():
    results = []
    config = Config(device=device)
    train_loader, val_loader, test_loader = load_umt_loaders(config, batch_size=BATCH_SIZE)


    print(f"\n====== Training with SGD (lr={LEARNING_RATE}, decay={DECAY}) ======")

    model = NNCRF(config).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG
    )

    writer = SummaryWriter(log_dir="runs/ner_synlstm")

    for epoch in range(1, EPOCHS + 1):
        adjust_lr(optimizer, epoch)
        train_loss, train_acc = train(
            model, optimizer, train_loader, device, epoch, config
        )
        val_loss, val_acc = validate(model, val_loader, device, config)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} | "
            f"Train NER Acc={train_acc['train_acc_ner']:.4f} F1={train_acc['train_f1_ner']:.4f} | "
            f"Train SA Acc={train_acc['train_acc_sa']:.4f} F1={train_acc['train_f1_sa']:.4f} || "
            f"Val NER Acc={val_acc['val_acc_ner']:.4f} F1={val_acc['val_f1_ner']:.4f} | "
            f"Val SA Acc={val_acc['val_acc_sa']:.4f} F1={val_acc['val_f1_sa']:.4f}"
        )


        writer.add_scalar("NER_Accuracy/train", train_acc["train_acc_ner"], epoch)
        writer.add_scalar("NER_F1/train", train_acc["train_f1_ner"], epoch)
        writer.add_scalar("SA_Accuracy/train", train_acc["train_acc_sa"], epoch)
        writer.add_scalar("SA_F1/train", train_acc["train_f1_sa"], epoch)

        writer.add_scalar("NER_Accuracy/val", val_acc["val_acc_ner"], epoch)
        writer.add_scalar("NER_F1/val", val_acc["val_f1_ner"], epoch)
        writer.add_scalar("SA_Accuracy/val", val_acc["val_acc_sa"], epoch)
        writer.add_scalar("SA_F1/val", val_acc["val_f1_sa"], epoch)


    save_model(model, config, "ner_model_synlstm")
    writer.close()
    results.append({"learning_rate": LEARNING_RATE, "val_acc": val_acc})

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/experiment_synlstm_results.csv", index=False)


if __name__ == "__main__":
    main()
