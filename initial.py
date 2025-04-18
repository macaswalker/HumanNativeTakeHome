import pandas as pd
from tqdm import tqdm
import time
from presidio_analyzer import AnalyzerEngine
from transformer_recognizer import TransformerRecognizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------- Setup ---------------- #
# Load data
df = pd.read_csv("Data_with_spans.csv")

# Supported entities based on your model's training
sample_entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "URL"]

# Set up the analyzer and register your custom recognizer
analyzer = AnalyzerEngine()
custom_recognizer = TransformerRecognizer(
    model_path="./pii_ner_model",
    supported_entities=sample_entities
)
analyzer.registry.add_recognizer(custom_recognizer)

# ---------------- Detection ---------------- #
start = time.time()
found_flags = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting PII with custom model"):
    text = row["value"]
    results = analyzer.analyze(text=text, entities=sample_entities, language="en")
    found_flags.append(int(len(results) > 0))

df["found_flag"] = found_flags
df.to_csv("Data_with_spans_and_found_flag.csv", index=False)
print(f"✅ Detection complete in {time.time() - start:.2f} seconds")

# ---------------- Evaluation ---------------- #
df = pd.read_csv("Data_with_spans_and_found_flag.csv")

# Overall report
print("\\n📊 Overall Classification Report:")
print(classification_report(df["flag"], df["found_flag"], target_names=["no_pii", "has_pii"]))

print("\\n🔄 Confusion Matrix:")
print(confusion_matrix(df["flag"], df["found_flag"]))

# Accuracy by pii_mode
print("\\n📁 Accuracy by PII Mode:\\n")
for mode in sorted(df["pii_mode"].unique()):
    subset = df[df["pii_mode"] == mode]
    y_true = subset["flag"]
    y_pred = subset["found_flag"]
    acc = accuracy_score(y_true, y_pred)
    print(f"{mode.title():<12}: Accuracy = {acc:.3f} ({len(subset)} samples)")

## inspect the data

## make sure everytime we run a classification with a different model we append a new flag instead of overwriting
## compare between the two models (sometimes - it is better to just use ootb )

## also time taken - this is 170 seconds for 1000 paragraphs at around 6 snetcnes each
