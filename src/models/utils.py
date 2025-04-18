tag2id = {
    'O': 0,
    'B-NAME': 1, 'I-NAME': 2,
    'B-EMAIL': 3, 'I-EMAIL': 4,
    'B-PHONE': 5, 'I-PHONE': 6,
    'B-LOCATION': 7, 'I-LOCATION': 8,
    'B-URL': 9, 'I-URL': 10,
}

id2tag = {v: k for k, v in tag2id.items()}