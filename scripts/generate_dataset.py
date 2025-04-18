import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_generation import build_dataset

if __name__ == "__main__":
    build_dataset()
