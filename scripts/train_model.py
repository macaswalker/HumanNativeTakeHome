import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

from src.models.utils import tag2id, id2tag
from src.training.dataset import PIITokenDataset, create_token_labels
from src.training.train import train_model

if __name__ == "__main__":
    # Load and prep data
    df = pd.read_csv("data/Data.csv")
    data = []

    # for each row in the datafram 
    for _, row in df.iterrows():
        if row['flag'] == 1: # if PII
            tokens, labels = create_token_labels(row['value'], row['pii_spans']) # tokenises and aligns with BIO
            data.append({'tokens': tokens, 'labels': labels}) # dict of tokens and labels

    for _, row in df[df['flag'] == 0].head(len(data) // 2).iterrows(): # esnure balanced dataset - not really needed here
        tokens = row['value'].split() # split on spaces
        labels = ['O'] * len(tokens) # all 0
        data.append({'tokens': tokens, 'labels': labels}) # append to list

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_ds = PIITokenDataset(train_data, tokenizer)
    val_ds = PIITokenDataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    # Model setup
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(tag2id)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train_model(model, train_loader, val_loader, device)

    # Save
    model.save_pretrained("pii_ner_model")
    tokenizer.save_pretrained("pii_ner_model")
