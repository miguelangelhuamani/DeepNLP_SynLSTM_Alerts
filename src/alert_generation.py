from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import warnings
from accelerate import Accelerator
from transformers import logging
from difflib import get_close_matches

class AlertGenerator:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device=None):
        self.accelerator = Accelerator()
        self.device = device if device else self.accelerator.device

        logging.set_verbosity_error()
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.iteration_times = 30

    def reconstruct_entities(self, tokens, tags):
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

    def validate_and_format_alert(self, raw_alert, entities, text):
        raw_alert = raw_alert.strip()
        
        #1: MODEL'S RESPONSE MAKES NO SENSE
        text_words = set(word.lower() for word in text.split())
        alert_words = set(word.lower() for word in raw_alert.split())
        
        if not alert_words.intersection(text_words):
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

        # 6. Redundant words in both sides (e.g., "Justin Bieber in Spain: Spain")
        theme_words = set(theme.lower().split())
        entity_words = set(entity.lower().split())
        
        if theme_words & entity_words:
            return None
    
        return f"{theme}: {entity}"

    def choose_best_alert(self, text, entities, alerts):  
        options_text = "\n".join([f"{i+1}. {alert}" for i, alert in enumerate(alerts)])
        entities_text = ", ".join(entities)
        
        prompt = (
            f"News article: \"{text}\"\n\n"
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,  
            do_sample=False,  
            top_p=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
        
        # Clean up the response further - remove any numbers and common prefixes
        cleaned_response = re.sub(r'^\s*\d+\.\s*', '', response).strip()
        
        # Try to find an exact match first
        if cleaned_response in alerts:
            return cleaned_response
        
        # If no exact match, use fuzzy matching with a less strict cutoff
        matches = get_close_matches(cleaned_response, alerts, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        # If all else fails, select the alert that contains the most entities from the text
        best_score = -1
        best_alert = alerts[0] if alerts else "NO_VALID_ALERT"
        
        for alert in alerts:
            score = sum(1 for entity in entities if entity.lower() in alert.lower())
            if score > best_score:
                best_score = score
                best_alert = alert
        
        return best_alert

    def generate_alert(self, text, entities, sentiment):
        entity_str = ", ".join(entities) if entities else "No notable entities"
        
        prompt = (
        f"News article: {text}\n"
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
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.9,
            temperature=0.6  
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        raw_alert = decoded.split("Alert: ")[-1].strip()
    
        if "\n" in raw_alert:
            raw_alert = raw_alert.split("\n")[0]
        
        return self.validate_and_format_alert(raw_alert, entities, text)

    def generate_multiple_alerts(self, text, tokens, tags, sentiment):
        entities = self.reconstruct_entities(tokens, tags)
        alerts = []
        
        for _ in range(self.iteration_times):
            alert = self.generate_alert(text, entities, sentiment)
            if alert != None:
                alerts.append(alert)

        best_alert = self.choose_best_alert(text, entities, alerts)
        return best_alert

