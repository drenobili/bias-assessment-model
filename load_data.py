# src/load_data.py

import sys
import os

# Add project root to Python path so config.py can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from datasets import load_dataset


def load_babe_dataset():
    """
    Load the BABE dataset from Hugging Face and split into train/test.

    The BABE dataset does not have a default validation split,
    so we split the training data 80/20.

    Returns:
        dataset: dict with 'train' and 'test' splits
    """
    dataset = load_dataset(config.DATASET_NAME)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    return dataset


if __name__ == "__main__":
    dataset = load_babe_dataset()
    print(dataset)
    print("\nSample entry:")
    print(dataset["train"][0])
