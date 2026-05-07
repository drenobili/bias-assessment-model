# src/evaluate.py

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from load_data import load_babe_dataset
from preprocess import create_dataloaders
import config

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)

model_path = os.path.join(PROJECT_ROOT, "models", "bert_babe.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model weights not found at {model_path}.\n"
        "Run src/train.py first to train and save the model."
    )

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_path}")


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set.
    Prints accuracy and a full classification report.
    """
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=["factual", "biased"]))


if __name__ == "__main__":
    dataset = load_babe_dataset()
    _, test_loader = create_dataloaders(dataset, batch_size=config.BATCH_SIZE)
    evaluate_model(model, test_loader)
