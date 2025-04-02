import pickle

# Mapeo de etiquetas de sentimiento
sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# Cargar los datos
with open("umt_data.pkl", "rb") as f:
    train_sents, train_labels, train_sa_labels, val_sents, val_labels, val_sa_labels, test_sents, test_labels, test_sa_labels = pickle.load(f)

# Contadores y ejemplos
counts = {0: 0, 1: 0, 2: 0}
examples = {0: [], 1: [], 2: []}

# Recorrer el dataset de entrenamiento
for sent, ner_tags, sentiment in zip(train_sents, train_labels, train_sa_labels):
    counts[sentiment] += 1
    if len(examples[sentiment]) < 5:
        examples[sentiment].append((sent, ner_tags, sentiment_map[sentiment]))

# Mostrar la cuenta total
print("🔢 Conteo total de etiquetas de sentimiento:")
for k, v in counts.items():
    print(f"{sentiment_map[k]}: {v} ejemplos")

# Mostrar los primeros 5 ejemplos por tipo
print("\n📋 Ejemplos (5 por categoría):\n")
for sentiment_value in [1, 2, 0]:  # NEUTRAL, POSITIVE, NEGATIVE
    print(f"🟦 Sentimiento: {sentiment_map[sentiment_value]}")
    for idx, (sent, ner_tags, label_name) in enumerate(examples[sentiment_value]):
        print(f"\nEjemplo {idx + 1}:")
        print("Tokens:", sent)
        print("NER Tags:", ner_tags)
        print("Sentiment Label:", label_name)
    print("-" * 60)
