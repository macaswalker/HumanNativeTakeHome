import json
import torch
from torch.utils.data import Dataset
from src.models.utils import tag2id


def create_token_labels(text, pii_spans):
    spans = json.loads(pii_spans) if isinstance(pii_spans, str) else pii_spans
    tokens = text.split()
    labels = ['O'] * len(tokens)

    token_boundaries = []
    current_pos = 0
    for token in tokens:
        start = text.find(token, current_pos)
        end = start + len(token)
        token_boundaries.append((start, end))
        current_pos = end

    for span in spans:
        s_start, s_end, s_type = span['start'], span['end'], span['type']
        for i, (t_start, t_end) in enumerate(token_boundaries):
            if t_start <= s_start < t_end:
                labels[i] = f'B-{s_type}'
            elif s_start <= t_start < s_end:
                labels[i] = f'I-{s_type}'

    return tokens, labels

class PIITokenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens']
        tags = self.data[idx]['labels']

        enc = self.tokenizer(
            tokens, is_split_into_words=True,
            truncation=True, padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = torch.ones(enc['input_ids'].shape, dtype=torch.long) * -100
        for i, word_idx in enumerate(enc.word_ids()):
            if word_idx is not None:
                if i > 0 and enc.word_ids()[i - 1] == word_idx:
                    continue
                labels[0, i] = tag2id.get(tags[word_idx], 0)

        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
