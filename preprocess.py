# src/preprocess.py

import sys
import os

# Add project root to Python path so config.py can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)


def tokenize_dataset(dataset_split):
    """
    Tokenize the 'text' column of a Hugging Face dataset split.

    Args:
        dataset_split: Hugging Face Dataset object (train or test split)

    Returns:
        TensorDataset with input_ids, attention_mask, and labels
    """
    texts = list(dataset_split["text"])
    labels = list(dataset_split["label"])

    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt"
    )

    labels_tensor = torch.tensor(labels)

    return TensorDataset(
        encoding["input_ids"],
        encoding["attention_mask"],
        labels_tensor
    )


def create_dataloaders(dataset, batch_size=config.BATCH_SIZE):
    """
    Create PyTorch DataLoaders for train and test splits.

    Args:
        dataset: dict with 'train' and 'test' keys (from load_babe_dataset)
        batch_size: int

    Returns:
        train_loader, test_loader
    """
    train_dataset = tokenize_dataset(dataset["train"])
    test_dataset = tokenize_dataset(dataset["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


# Optional: run directly to verify shapes
if __name__ == "__main__":
    from load_data import load_babe_dataset  # fixed import
    dataset = load_babe_dataset()
    train_loader, test_loader = create_dataloaders(dataset)

    batch = next(iter(train_loader))
    print("input_ids shape    :", batch[0].shape)
    print("attention_mask shape:", batch[1].shape)
    print("labels shape       :", batch[2].shape)
