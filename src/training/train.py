import torch
from torch.optim import AdamW
from tqdm import tqdm
from src.models.utils import tag2id, id2tag


def train_model(model, train_loader, val_loader, device, num_epochs=3):
    model.to(device)
    opt = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = out.loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        print(f"\nTrain Loss: {train_loss / len(train_loader):.4f}")

        evaluate(model, val_loader, device)

def evaluate(model, val_loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=mask).logits
            pred = torch.argmax(logits, dim=2)

            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        preds.append(pred[i, j].item())
                        truths.append(labels[i, j].item())

    acc = sum(p == t for p, t in zip(preds, truths)) / len(preds)
    print(f"Val Accuracy: {acc:.4f}")

    per_entity(preds, truths)

def per_entity(preds, truths):
    correct, total = {}, {}
    for p, t in zip(preds, truths):
        true_tag = id2tag[t]
        pred_tag = id2tag[p]

        if true_tag == 'O':
            continue

        entity = true_tag.split('-')[1]
        total[entity] = total.get(entity, 0) + 1
        if pred_tag == true_tag:
            correct[entity] = correct.get(entity, 0) + 1

    print("\nPer-entity accuracy:")
    for ent in total:
        acc = correct.get(ent, 0) / total[ent]
        print(f"{ent:<10}: {acc:.4f} ({correct.get(ent, 0)}/{total[ent]})")
