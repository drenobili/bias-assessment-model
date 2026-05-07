# Bias Detection and Assessment Using BERT

A CS thesis project that fine-tunes BERT to detect and quantify media bias in written text. The system classifies text as **biased or factual**, produces a **continuous bias intensity score**, and identifies the **specific words** most responsible for the prediction using gradient-based attribution.

---

## Overview

| Output | Description |
|---|---|
| **Binary Label** | `biased` or `factual` |
| **Bias Score** | Continuous value from 0.0 to 1.0 |
| **Top Biased Words** | Tokens ranked by gradient importance (biased texts only) |

This project extends beyond simple classification by addressing:
- **Bias quantification** — not just detection
- **Explainable AI** — why did the model flag this text?
- **Responsible NLP** — transparent, interpretable predictions

---

## Dataset

**BABE — Bias Annotations By Experts**
- Source: [`mediabiasgroup/BABE`](https://huggingface.co/datasets/mediabiasgroup/BABE) on Hugging Face
- Contains news sentences annotated for media bias and subjective framing
- Split: 80% train / 20% test (seeded for reproducibility)

The dataset is downloaded automatically at runtime via the Hugging Face `datasets` library. No manual download required.

---

## Project Structure

```
├── config.py              # All hyperparameters and path settings
├── requirements.txt       # Python dependencies
├── src/
│   ├── load_data.py       # Downloads and splits the BABE dataset
│   ├── preprocess.py      # Tokenization and DataLoader creation
│   ├── train.py           # Fine-tuning loop with per-epoch evaluation
│   ├── evaluate.py        # Full classification report on test set
│   └── predict.py         # Single-text prediction + interactive CLI
└── models/                # Saved model weights (not tracked by Git)
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

All scripts are run from the **project root**, not from inside `src/`.

### Train the model
```bash
python src/train.py
```
Trains BERT for 3 epochs and saves weights to `models/bert_babe.pt`.

### Evaluate on the test set
```bash
python src/evaluate.py
```
Prints accuracy, precision, recall, and F1-score.

### Run interactive prediction
```bash
python src/predict.py
```
Loads the trained model and accepts text input in a loop. Type `exit` to quit.

**Example output:**
```
Enter news text: The radical left is pushing an extreme agenda to destroy our values.

Prediction    : biased
Bias Score    : 0.9312
Factual Score : 0.0688
Interpretation: Strongly biased

Top Biased Words:
  radical              (3.8421)
  extreme              (3.1205)
  destroy              (2.9034)
  pushing              (1.7823)
  agenda               (1.6541)
```

---

## Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `bert-base-uncased` | Pretrained model from Hugging Face |
| `DATASET_NAME` | `mediabiasgroup/BABE` | Dataset identifier |
| `MAX_LENGTH` | `128` | Max token length per input |
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `3` | Fine-tuning epochs |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `WEIGHT_DECAY` | `0.01` | L2 regularization |

---

## Model Architecture

- **Base model:** `bert-base-uncased` (110M parameters)
- **Task head:** `BertForSequenceClassification` with 2 output labels
- **Optimizer:** AdamW with linear warmup scheduler
- **Loss:** Cross-entropy
- **Explainability:** Gradient-based token attribution (L2 norm of embedding gradients)

---

## Notes on Model Weights

Trained model weights (`models/bert_babe.pt`) are **not included** in this repository due to file size. To use the system, run `src/train.py` first to generate the weights locally.

---

## Research Context

This thesis explores how NLP and deep learning can be used to:
- Detect and quantify biased language in news and public discourse
- Produce human-understandable explanations for model predictions
- Contribute to AI ethics, fairness-aware modeling, and misinformation detection

**Techniques used:** Transfer learning, fine-tuning, gradient-based attribution, explainable AI (XAI)

---

## Author

**Andre Bautista**
BS Computer Science — University of the East - Caloocan
