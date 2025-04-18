import random
from faker import Faker

faker = Faker()

# Define PII injectors
PII_TYPES = {
    "NAME": faker.name,
    "EMAIL": faker.email,
    "LOCATION": faker.city,
    "PHONE": faker.phone_number,
    "URL": lambda: f"https://{faker.domain_name()}"
}

EMBEDDED_TEMPLATES = {
    "NAME": lambda: f"My name is {faker.name()}",
    "EMAIL": lambda: f"You can email me at {faker.email()}",
    "LOCATION": lambda: f"I live in {faker.city()}",
    "PHONE": lambda: f"My number is {faker.phone_number()}",
    "URL": lambda: f"Visit my site at https://{faker.domain_name()}"
}

def generate_base_paragraph():
    return faker.paragraph(nb_sentences=6)

def inject_pii(text, pii_type, mode):
    pii_value = PII_TYPES[pii_type]()

    if mode == "standalone":
        return f"{pii_value}. {text}", pii_value

    elif mode == "raw":
        return f"{text}. {pii_value}", pii_value

    elif mode == "embedded":

        if pii_type == "EMAIL":
            embedded = f"You can email me at {pii_value}"
        elif pii_type == "NAME":
            embedded = f"My name is {pii_value}"
        elif pii_type == "LOCATION":
            embedded = f"I live in {pii_value}"
        elif pii_type == "PHONE":
            embedded = f"My number is {pii_value}"
        elif pii_type == "URL":
            embedded = f"Visit my site at {pii_value}"
        else:
            embedded = pii_value

        insert_point = random.choice([0, 1])
        sentences = text.split(". ")
        if insert_point == 0:
            return f"{embedded}. " + ". ".join(sentences), pii_value
        else:
            sentences[-1] += f". {embedded}"
            return ". ".join(sentences), pii_value

    else: 
        return text, None


def get_span(text, pii_value, pii_type):

    if pii_value and pii_value in text:
        start = text.index(pii_value)
        end = start + len(pii_value)
        return {
            "type": pii_type,
            "start": start,
            "end": end,
            "value": pii_value
        }
    return None

def generate_entry(pii_mode="embedded"):

    text = generate_base_paragraph()
    flag = int(pii_mode != "none")
    pii_metadata = []

    if flag:
        pii_type = random.choice(list(PII_TYPES))
        text, pii_value = inject_pii(text, pii_type, pii_mode)
        span = get_span(text, pii_value, pii_type)
        if span:
            pii_metadata.append(span)

    return {
        "text": text,
        "flag": flag,
        "pii_mode": pii_mode,
        "pii_spans": pii_metadata
    }

if __name__ == "__main__":
    entry = generate_entry("embedded")
    print(entry["text"])
    print(entry["pii_spans"])