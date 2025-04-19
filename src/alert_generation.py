from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import pickle
import os
import re

from accelerate import Accelerator
from utils import load_conll3_data

accelerator = Accelerator()

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

import warnings
from transformers import logging

# Suppress the warnings from transformers
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

model.config.pad_token_id = model.config.eos_token_id

def load_conll3_data(file_path):
    sentences = []
    labels = []
    sentiment = []
    
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split('\t')
                current_sentence.append(parts[0])
                current_labels.append(parts[1])
                if len(parts) > 2:
                    if len(sentiment) < len(sentences) + 1:
                        sentiment.append(int(parts[2]))
    
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
    if not sentiment:
        sentiment = [1] * len(sentences)
    
    return sentences, labels, sentiment

def cargar_datos_umt():
    data_path = "data/umt_data.pkl"
    if not os.path.exists(data_path):
        raise FileNotFoundError("umt_data.pkl no encontrado. Asegúrate de haberlo generado.")
    
    with open(data_path, "rb") as f:
        return pickle.load(f)

def reconstruct_tweet(tokens):
    return " ".join(tokens)

def reconstruct_entities(tokens, tags):
    entities = []
    current = []
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                entities.append(" ".join(current))
            current = [token]
        elif tag.startswith("I-") and current:
            current.append(token)
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    return entities

def sentiment_from_label(label_num):
    """
    Converts a numeric sentiment label to text.
    0 = negative, 1 = neutral, 2 = positive
    """
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    return labels.get(label_num, "neutral")  # Default to neutral if not in the dictionary


def validate_and_format_alert(raw_alert, entities, tweet):
    raw_alert = raw_alert.strip()
    
    #1: MODEL'S RESPONSE MAKES NO SENSE
    tweet_words = set(word.lower() for word in tweet.split())
    alert_words = set(word.lower() for word in raw_alert.split())
    
    if not alert_words.intersection(tweet_words):
        return None
    
    # 2. Check "theme: entity" format
    if ":" not in raw_alert:
        return None

    theme, entity = map(str.strip, raw_alert.split(":", 1))
    
    # 3. Ensure theme and entity are both non-empty
    if not theme or not entity:
        return None

    # 4. Entity must match one of the known entities (if provided)
    if entities and not any(ent.lower() in raw_alert.lower() for ent in entities):
        return None

    # 5. Eliminate redundant alerts (e.g., "Justin Bieber: Justin Bieber")
    if theme.lower() == entity.lower():
        return None

    return f"{theme}: {entity}"

def generate_multiple_alerts(tweet, entities, sentiment, num_intentos=30):
    """
    Genera múltiples alertas y selecciona la mejor según criterios de calidad.
    """
    alertas = []
    
    for _ in range(num_intentos):
        alerta = generate_alert(tweet, entities, sentiment)
        if alerta != None:
            #print("Alerta ACTUAL -", alerta)
            alertas.append(alerta)
    
    mejor_alerta = choose_best_alert(tweet, entities, alertas)
    return mejor_alerta


def choose_best_alert(tweet: str, entities: list[str], alerts: list[str]):
    options_text = "\n".join([f"{i+1}. {alert}" for i, alert in enumerate(alerts)])
    
    entities_text = ", ".join(entities)
    
    prompt = (
        f"News article: \"{tweet}\"\n\n"
        f"Important entities in this news article: {entities_text}\n\n"
        f"Alert options:\n{options_text}\n\n"
        f"Instructions: Select the BEST alert from the options above that:\n"
        f"1. Accurately summarizes the key message of the news article\n"
        f"2. Includes the most relevant entity or location mentioned\n"
        f"3. Provides useful context for understanding the news article's content\n"
        f"4. Is clear and concise\n\n"
        f"First, analyze each option carefully. Then respond ONLY with the number and exact text of the best alert.\n"
        f"Example correct response format: \"2. Traffic alert: I-90\"\n\n"
        f"Selected alert:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  
        do_sample=False,  
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = decoded.split("Selected alert:")[-1].strip()
    
    # Method 1: Check if the response starts with a number followed by a period
    number_match = re.match(r'^\s*(\d+)\.\s*(.*)', response)
    
    if number_match:
        try:
            index = int(number_match.group(1)) - 1
            if 0 <= index < len(alerts):
                return alerts[index]
        except (ValueError, IndexError):
            pass
    
    # Method 2: Look for the exact match or closest match
    from difflib import get_close_matches
    
    # Clean up the response further - remove any numbers and common prefixes
    cleaned_response = re.sub(r'^\s*\d+\.\s*', '', response).strip()
    
    # Try to find an exact match first
    if cleaned_response in alerts:
        return cleaned_response
    
    # If no exact match, use fuzzy matching with a less strict cutoff
    matches = get_close_matches(cleaned_response, alerts, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    # If all else fails, select the alert that contains the most entities from the tweet
    best_score = -1
    best_alert = alerts[0] if alerts else "NO_VALID_ALERT"
    
    for alert in alerts:
        score = sum(1 for entity in entities if entity.lower() in alert.lower())
        if score > best_score:
            best_score = score
            best_alert = alert
    
    return best_alert


def generate_alert(tweet, entities, sentiment):
    entity_str = ", ".join(entities) if entities else "No notable entities"
    
    prompt = (
    f"News article: {tweet}\n"
    f"Entities involved: {entity_str}\n"
    f"Sentiment: {sentiment}\n\n"
    f"Task: Create a brief alert in the format 'theme: entity' where:\n"
    f"- 'theme' is a 1-3 word description of what's happening (avoid generic terms like 'Update' or 'News')\n"
    f"- 'entity' is one of the entities mentioned in the article\n"
    f"- The entity MUST be from the list: {entity_str}\n"
    f"- The theme should reflect the specific content and sentiment of the article\n\n"
    f"Examples:\n"
    f"News article: The European Union has announced new health measures in response to mad cow disease.\n"
    f"Entities: European Union, mad cow disease\n"
    f"Sentiment: neutral\n"
    f"Alert: Health measures: European Union\n\n"
    f"News article: German Chancellor Angela Merkel visited Paris to discuss climate change initiatives.\n"
    f"Entities: Angela Merkel, Paris, climate change\n"
    f"Sentiment: positive\n"
    f"Alert: Diplomatic visit: Angela Merkel\n\n"
    f"News article: The International Monetary Fund warns of a potential recession in 2024.\n"
    f"Entities: International Monetary Fund, recession, 2024\n"
    f"Sentiment: negative\n"
    f"Alert: Economic warning: International Monetary Fund\n\n"
    f"News article: The United Nations held a conference to address global water scarcity.\n"
    f"Entities: United Nations, global water scarcity\n"
    f"Sentiment: neutral\n"
    f"Alert: Conference on water scarcity: United Nations\n\n"
    f"News article: Flooding in New Orleans has caused widespread damage after Hurricane Ida.\n"
    f"Entities: New Orleans, Hurricane Ida\n"
    f"Sentiment: negative\n"
    f"Alert: Natural disaster: New Orleans\n\n"
    f"Now, let's think through this step by step:\n"
    f"1. What is the main action or event in the article? (e.g., announcement, visit, warning, disaster)\n"
    f"2. Which entity from {entity_str} is most relevant to this action?\n"
    f"3. Create a 1-3 word theme based on the action (avoid generic words like 'Update' or 'News')\n"
    f"4. Format the alert as 'theme: entity'\n\n"
    f"Now generate the alert for the news article about {entities[0] if entities else 'the topic'}:\n"
    f"Alert: ")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        top_p=0.9,
        temperature=0.6  
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    raw_alert = decoded.split("Alert: ")[-1].strip()

    if "\n" in raw_alert:
        raw_alert = raw_alert.split("\n")[0]
    
    return validate_and_format_alert(raw_alert, entities, tweet)

def main():
    
    train_sentences, train_labels, train_sa = load_conll3_data(file_path="data/conll3/train.txt")
    
    for i in range(0,50):  
        tokens = train_sentences[i]
        tags = train_labels[i]
        sen = train_sa[i]
        
        tweet = reconstruct_tweet(tokens)
        entities = reconstruct_entities(tokens, tags)
        sentiment = sentiment_from_label(sen)

        alerta = generate_multiple_alerts(tweet, entities, sentiment)
        alerta = alerta.split("\n")[0].strip()
        print("Model is on device:", model.device)
        print(f"[{i+1}] News article: {tweet}")
        print(f"     Entities: {entities}")
        print(f"     Sentiment: {sentiment}")
        print(f"     Generated alert: {alerta}\n")
        

if __name__ == "__main__":
    main()