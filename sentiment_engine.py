import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Windows Path Fix
try:
    torch.classes.__path__ = []
except Exception:
    pass

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print("🧠 Loading AI Brain...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error: {e}")

    def analyze(self, text):
        try:
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            
            ranking = np.argsort(scores)
            top_rank = ranking[-1]
            
            if top_rank == 0:
                label = "Negative"
                compound = -1 * scores[0] 
            elif top_rank == 2:
                label = "Positive"
                compound = scores[2]
            else:
                label = "Neutral"
                compound = 0.0
                
            # Slang Override
            if "fuck yeah" in text.lower():
                label = "Positive"
                compound = 0.95

            return {"score": float(compound), "label": label}

        except Exception:
            return {"score": 0.0, "label": "Neutral"}

    def generate_session_report(self, scores_list):
        """
        FINAL REQUIREMENT: Summarise trend or shift in mood.
        """
        if not scores_list:
            return {"verdict": "No Data", "trend": "None", "final_score": 0}
        
        # 1. Calculate Average (Tier 1)
        avg_score = sum(scores_list) / len(scores_list)
        if avg_score >= 0.2: verdict = "Positive (Satisfied)"
        elif avg_score <= -0.2: verdict = "Negative (Unsatisfied)"
        else: verdict = "Neutral (Balanced)"
        
        # 2. Calculate Shift/Trend (Bonus Tier)
        # We compare the start of the conversation to the end
        if len(scores_list) < 2:
            trend = "Not enough data to detect shift."
        else:
            # Take avg of first 2 vs last 2 to dampen noise
            window = max(1, len(scores_list)//3)
            start_avg = sum(scores_list[:window]) / window
            end_avg = sum(scores_list[-window:]) / window
            
            diff = end_avg - start_avg
            
            # Refined Logic for specific "Happy -> Angry" detection
            if diff > 0.3:
                trend = "📈 POSITIVE SHIFT (Frustrated → Satisfied)"
            elif diff < -0.3:
                trend = "📉 NEGATIVE SHIFT (Happy → Frustrated)"
            else:
                trend = "➡️ STABLE (Mood remained consistent)"
                
        return {
            "verdict": verdict,
            "trend": trend,
            "final_score": avg_score
        }
