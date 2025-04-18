import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_generation import build_dataset

if __name__ == "__main__":
    build_dataset(
        output_path_csv="data/test/test_Data.csv",
        output_path_json="data/test/test_Data.json",
        seed=1337
    )
