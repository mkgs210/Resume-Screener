from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class ResumeBasedPredictionModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model(
        self,
        base_model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        path_to_trained_model: str | Path,
    ) -> "ResumeBasedPredictionModel":
        self.tokenizer = tokenizer
        self.model = PeftModel.from_pretrained(base_model, path_to_trained_model)
        return self

    def predict_interview_outcome(self, text: str) -> int:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1).item()

        if not isinstance(predictions, int):
            predictions = int(predictions)

        return predictions
