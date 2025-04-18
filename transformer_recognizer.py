from presidio_analyzer import EntityRecognizer, RecognizerResult
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

from presidio_analyzer import EntityRecognizer, RecognizerResult
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class TransformerRecognizer(EntityRecognizer):
    def __init__(self, model_path: str, supported_entities=None, device=None):
        super().__init__(supported_entities=supported_entities or [])

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Use label mapping from model config
        self.label2entity = self.model.config.id2label
        self.entity2label = self.model.config.label2id

    def analyze(self, text: str, entities=None, nlp_artifacts=None):
        tokens = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True
        )

        offset_mapping = tokens.pop("offset_mapping")  # important!
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)

        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        scores = torch.softmax(outputs.logits, dim=2).squeeze()
        input_ids = tokens["input_ids"].squeeze().tolist()
        offsets = offset_mapping.squeeze().tolist()

        results = []
        for idx, pred_id in enumerate(predictions):
            label = self.label2entity.get(str(pred_id), "O")
            if label == "O" or label == "LABEL_0":
                continue

            start, end = offsets[idx]
            if start == end:
                continue

            score = scores[idx][pred_id].item()
            entity_type = label.replace("B-", "").replace("I-", "")

            if entities and entity_type not in entities:
                continue

            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=score
                )
            )

        return results


from presidio_analyzer import AnalyzerEngine

# Your supported entities (must match your model's label scheme)
supported_entities = ["PERSON", "EMAIL_ADDRESS", "LOCATION", "PHONE_NUMBER", "URL"]

# Setup Presidio and register your recognizer
analyzer = AnalyzerEngine()
custom_recognizer = TransformerRecognizer(
    model_path="./pii_ner_model", 
    supported_entities=supported_entities
)
analyzer.registry.add_recognizer(custom_recognizer)


text = "Hi, I'm Mac Walker. You can reach me at macskyewalker@gmail.com or call 123-456-7890."

results = analyzer.analyze(text=text, language="en")
for res in results:
    print(res)


