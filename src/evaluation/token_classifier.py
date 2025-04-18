import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from src.models.utils import id2tag


def load_bert_token_model(model_path="pii_ner_model"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForTokenClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_token_spans(text, tokenizer, model):
    inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offset_mapping = inputs.pop("offset_mapping")
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2).squeeze()
    offsets = offset_mapping.squeeze().tolist()

    spans = []
    for pred_id, (start, end) in zip(predictions, offsets):
        label = id2tag.get(pred_id.item(), "O")
        if label != "O" and start != end:
            spans.append((label, text[start:end]))

    return spans


def evaluate_on_dataframe(df, tokenizer, model, column="value", flag_name="token_model_flag"):
    from tqdm import tqdm
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Token Classifier"):
        try:
            spans = predict_token_spans(row[column], tokenizer, model)
            results.append(int(len(spans) > 0))
        except Exception as e:
            print(f"Error: {e}")
            results.append(0)

    df[flag_name] = results
    return df
