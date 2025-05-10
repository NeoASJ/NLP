import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

class SentimentAnalyzer:
    def __init__(self, model_name: str = "siebert/sentiment-roberta-large-english"):
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
       
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    def analyze(self, text: str | List[str]) -> List[Dict]:
        
        results = self.classifier(text)
        return results


analyzer = SentimentAnalyzer()
texts = [
        "This movie was terrible.",
        "The service was excellent!",
        "Meh, it was okay."
]
print(analyzer.analyze(texts))
