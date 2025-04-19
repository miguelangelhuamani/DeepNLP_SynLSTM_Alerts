import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from src.utils import load_umt_data


def sentiment_name(sentiment_id):
    """Convert sentiment ID to readable name"""
    return "Positive" if sentiment_id == 1 else "Negative"


def analyze_dataset():
    # Load the data
    print("Cargando dataset...")
    (
        train_sentences,
        train_labels,
        train_sa,
        val_sentences,
        val_labels,
        val_sa,
        test_sentences,
        test_labels,
        test_sa,
    ) = load_umt_data()

    # Crear directorio para guardar estadísticas
    os.makedirs("stats", exist_ok=True)

    # ======= ANÁLISIS DE LONGITUD =======

    # Calcular longitudes
    train_lengths = [len(s) for s in train_sentences]
    val_lengths = [len(s) for s in val_sentences]
    test_lengths = [len(s) for s in test_sentences]

    # Estadísticas de longitud
    train_mean = np.mean(train_lengths)
    train_median = np.median(train_lengths)
    train_max = np.max(train_lengths)
    val_mean = np.mean(val_lengths)
    val_median = np.median(val_lengths)
    val_max = np.max(val_lengths)
    test_mean = np.mean(test_lengths)
    test_median = np.median(test_lengths)
    test_max = np.max(test_lengths)

    # 1. Histograma comparativo
    plt.figure(figsize=(12, 6))
    plt.hist(train_lengths, alpha=0.6, label="Training", bins=30, color="blue")
    plt.hist(val_lengths, alpha=0.6, label="Validation", bins=30, color="green")
    plt.hist(test_lengths, alpha=0.6, label="Testing", bins=30, color="red")
    plt.xlabel("Longitud de texto (tokens)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.title("Distribución de longitudes en datasets")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("stats/longitud_comparativa.png")

    # 2. Histogramas individuales
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Training set
    axes[0].hist(train_lengths, bins=30, color="blue", alpha=0.7)
    axes[0].axvline(
        train_mean,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Media: {train_mean:.1f}",
    )
    axes[0].axvline(
        train_median,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Mediana: {train_median:.1f}",
    )
    axes[0].set_title(f"Longitud en Training Set (máx: {train_max})")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Validation set
    axes[1].hist(val_lengths, bins=30, color="green", alpha=0.7)
    axes[1].axvline(
        val_mean,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Media: {val_mean:.1f}",
    )
    axes[1].axvline(
        val_median,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Mediana: {val_median:.1f}",
    )
    axes[1].set_title(f"Longitud en Validation Set (máx: {val_max})")
    axes[1].set_ylabel("Frecuencia")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Test set
    axes[2].hist(test_lengths, bins=30, color="red", alpha=0.7)
    axes[2].axvline(
        test_mean,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Media: {test_mean:.1f}",
    )
    axes[2].axvline(
        test_median,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Mediana: {test_median:.1f}",
    )
    axes[2].set_title(f"Longitud en Test Set (máx: {test_max})")
    axes[2].set_xlabel("Longitud de texto (tokens)")
    axes[2].set_ylabel("Frecuencia")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("stats/longitudes_individuales.png")

    # ======= ANÁLISIS DE SENTIMIENTO =======

    # Conteo de ejemplos por conjunto
    train_count = len(train_sentences)
    val_count = len(val_sentences)
    test_count = len(test_sentences)
    total_count = train_count + val_count + test_count

    # Conteo de sentimientos
    train_sentiments = Counter(train_sa)
    val_sentiments = Counter(val_sa)
    test_sentiments = Counter(test_sa)

    # Combinación de todos los sentimientos
    all_sentiments = Counter(train_sa + val_sa + test_sa)

    # Crear gráfico de distribución de sentimientos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colores para los sentimientos
    colors = ["#FF6B6B", "#59CD90"]
    labels = ["Negativo", "Positivo"]

    # Training set
    train_values = [train_sentiments.get(0, 0), train_sentiments.get(1, 0)]
    axes[0, 0].pie(
        train_values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    axes[0, 0].set_title(f"Sentimientos en Training ({train_count} ejemplos)")

    # Validation set
    val_values = [val_sentiments.get(0, 0), val_sentiments.get(1, 0)]
    axes[0, 1].pie(
        val_values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    axes[0, 1].set_title(f"Sentimientos en Validation ({val_count} ejemplos)")

    # Test set
    test_values = [test_sentiments.get(0, 0), test_sentiments.get(1, 0)]
    axes[1, 0].pie(
        test_values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    axes[1, 0].set_title(f"Sentimientos en Test ({test_count} ejemplos)")

    # Todos los conjuntos
    all_values = [all_sentiments.get(0, 0), all_sentiments.get(1, 0)]
    axes[1, 1].pie(
        all_values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    axes[1, 1].set_title(f"Sentimientos en Total ({total_count} ejemplos)")

    plt.tight_layout()
    plt.savefig("stats/distribucion_sentimientos.png")

    # ======= ANÁLISIS DE ETIQUETAS NER =======

    # Extraer etiquetas NER de cada conjunto
    train_ner_tags = [tag for sent_tags in train_labels for tag in sent_tags]
    val_ner_tags = [tag for sent_tags in val_labels for tag in sent_tags]
    test_ner_tags = [tag for sent_tags in test_labels for tag in sent_tags]
    all_ner_tags = train_ner_tags + val_ner_tags + test_ner_tags

    # Contar frecuencia de etiquetas
    train_tag_counts = Counter(train_ner_tags)
    val_tag_counts = Counter(val_ner_tags)
    test_tag_counts = Counter(test_ner_tags)
    all_tag_counts = Counter(all_ner_tags)

    # Extraer etiquetas únicas (excluyendo O)
    unique_tags = sorted([tag for tag in all_tag_counts.keys() if tag != "O"])

    # Crear gráfico de barras para distribución de entidades
    plt.figure(figsize=(14, 8))

    # Preparar datos para gráfico
    x = np.arange(len(unique_tags))
    width = 0.2

    # Barras para cada conjunto
    train_bars = [train_tag_counts.get(tag, 0) for tag in unique_tags]
    val_bars = [val_tag_counts.get(tag, 0) for tag in unique_tags]
    test_bars = [test_tag_counts.get(tag, 0) for tag in unique_tags]

    # Crear gráfico
    plt.bar(x - width, train_bars, width, label="Training", color="blue", alpha=0.7)
    plt.bar(x, val_bars, width, label="Validation", color="green", alpha=0.7)
    plt.bar(x + width, test_bars, width, label="Testing", color="red", alpha=0.7)

    plt.xlabel("Etiqueta NER")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de etiquetas NER por conjunto de datos")
    plt.xticks(x, unique_tags, rotation=45)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("stats/distribucion_ner.png")

    # ======= MOSTRAR ESTADÍSTICAS POR TERMINAL =======
    print("\n" + "=" * 50)
    print("ANÁLISIS DEL DATASET")
    print("=" * 50)

    print("\n📊 NÚMERO DE FRASES:")
    print(f"  • Training:   {train_count}")
    print(f"  • Validation: {val_count}")
    print(f"  • Testing:    {test_count}")
    print(f"  • TOTAL:      {total_count}")

    print("\n📏 LONGITUD DE FRASES (tokens):")
    print(
        f"  • Training:   media={train_mean:.2f}, mediana={train_median:.1f}, máximo={train_max}"
    )
    print(
        f"  • Validation: media={val_mean:.2f}, mediana={val_median:.1f}, máximo={val_max}"
    )
    print(
        f"  • Testing:    media={test_mean:.2f}, mediana={test_median:.1f}, máximo={test_max}"
    )

    print("\n🔍 DISTRIBUCIÓN DE SENTIMIENTOS:")
    print("  • TRAINING:")
    for sentiment_id, count in sorted(train_sentiments.items()):
        print(
            f"    - {sentiment_name(sentiment_id):8}: {count:5} ({count/train_count*100:6.2f}%)"
        )

    print("  • VALIDATION:")
    for sentiment_id, count in sorted(val_sentiments.items()):
        print(
            f"    - {sentiment_name(sentiment_id):8}: {count:5} ({count/val_count*100:6.2f}%)"
        )

    print("  • TESTING:")
    for sentiment_id, count in sorted(test_sentiments.items()):
        print(
            f"    - {sentiment_name(sentiment_id):8}: {count:5} ({count/test_count*100:6.2f}%)"
        )

    print("\n  • TOTAL:")
    for sentiment_id, count in sorted(all_sentiments.items()):
        print(
            f"    - {sentiment_name(sentiment_id):8}: {count:5} ({count/total_count*100:6.2f}%)"
        )

    print("\n🔖 ETIQUETAS NER:")
    # Mostrar etiquetas y sus conteos en todos los conjuntos
    print(
        f"  • Número de tipos de etiquetas únicas: {len(unique_tags) + 1} (incluyendo 'O')"
    )
    print(f"  • Etiquetas encontradas: {', '.join(['O'] + unique_tags)}")
    print("\n  • Distribución de etiquetas (excluyendo 'O'):")

    for tag in unique_tags:
        print(
            f"    - {tag:6}: Train={train_tag_counts.get(tag, 0):5}, Val={val_tag_counts.get(tag, 0):5}, Test={test_tag_counts.get(tag, 0):5}, Total={all_tag_counts.get(tag, 0):5}"
        )

    print(
        f"\n    - {'O':6}: Train={train_tag_counts.get('O', 0):5}, Val={val_tag_counts.get('O', 0):5}, Test={test_tag_counts.get('O', 0):5}, Total={all_tag_counts.get('O', 0):5}"
    )

    # Información sobre archivos generados
    print("\n📁 ARCHIVOS GENERADOS:")
    print("  • stats/longitud_comparativa.png - Histograma comparativo de longitudes")
    print(
        "  • stats/longitudes_individuales.png - Histogramas individuales de longitud"
    )
    print("  • stats/distribucion_sentimientos.png - Distribución de sentimientos")
    print("  • stats/distribucion_ner.png - Distribución de etiquetas NER")


if __name__ == "__main__":
    analyze_dataset()
