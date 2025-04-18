import pandas as pd
import random
import uuid
import json
from tqdm import tqdm
from src.data.data_generation import generate_entry

def build_dataset(
    output_path_csv="data/Data.csv",
    output_path_json="data/Data.json",
    modes=("embedded", "raw", "standalone", "none"),
    examples_per_mode=250,
    seed=42
):
    """
    Build a full dataset of synthetic blog posts with PII and save to disk.
    
    Args:
        output_path_csv (str): path to save CSV version
        output_path_json (str): path to save JSON lines version
        modes (tuple): PII modes to include
        examples_per_mode (int): number of samples per mode
        seed (int): seed for reproducibility
    """

    random.seed(seed)
    dataset_id = str(uuid.uuid4())
    rows = []

    for mode in modes:
        for _ in tqdm(range(examples_per_mode), desc=f"Generating mode: {mode}"):
            entry = generate_entry(pii_mode=mode)
            rows.append({
                "dataset_id": dataset_id,
                "data_id": str(uuid.uuid4()),
                "value": entry["text"],
                "flag": entry["flag"],
                "pii_mode": entry["pii_mode"],
                "pii_spans": json.dumps(entry["pii_spans"])
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(output_path_csv, index=False)
    df.to_json(output_path_json, orient="records", lines=True)

    print(f"Dataset saved to {output_path_csv} and {output_path_json}")
    return df

