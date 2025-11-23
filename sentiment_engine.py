import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# --- THE FIX YOU FOUND (Apply immediately after imports) ---
try:
    torch.classes.__path__ = []
except Exception:
    pass
# ---------------------------------------------------------

class SentimentAnalyzer:
    def __init__(self):
        # Using the advanced Transformer model
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        print("🧠 Loading AI Brain... (This might take 30s the first time)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print("✅ AI Brain Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def analyze(self, text):
        """
        Analyzes text using RoBERTa.
        """
        try:
            # 1. Run the AI Model
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            
            # 2. Convert raw math (logits) to probabilities
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            
            # Ranking: 0=Negative, 1=Neutral, 2=Positive
            ranking = np.argsort(scores)
            top_rank = ranking[-1]
            
            if top_rank == 0:
                label = "Negative"
                # Invert score for the graph (so it looks red/downward)
                compound = -1 * scores[0] 
            elif top_rank == 2:
                label = "Positive"
                compound = scores[2]
            else:
                label = "Neutral"
                compound = 0.0
            
            # Bonus: Handle "Strong" slang manually if the model is unsure
            # (Optional hybrid approach)
            if "fuck yeah" in text.lower() and label == "Neutral":
                label = "Positive"
                compound = 0.9

            return {"score": float(compound), "label": label}

        except Exception as e:
            print(f"Analysis Error: {e}")
            # Fallback to Neutral if AI crashes
            return {"score": 0.0, "label": "Neutral"}

    def get_overall_verdict(self, scores_list):
        if not scores_list: return "Neutral"
        avg_score = sum(scores_list) / len(scores_list)
        if avg_score >= 0.2: return "Positive - Customer Satisfied"
        if avg_score <= -0.2: return "Negative - Issues Unresolved"
        return "Neutral - Standard Interaction"