# config.py

# ---------------- Model and Dataset ----------------
MODEL_NAME = "bert-base-uncased"       # Pretrained BERT model from Hugging Face
DATASET_NAME = "mediabiasgroup/BABE"   # BABE dataset on Hugging Face Hub

# ---------------- Training Parameters ----------------
MAX_LENGTH = 128        # Max token length per input (BERT limit is 512)
BATCH_SIZE = 16         # Batch size for training and evaluation
EPOCHS = 3              # Number of fine-tuning epochs
LEARNING_RATE = 2e-5    # AdamW learning rate (standard for BERT fine-tuning)
WEIGHT_DECAY = 0.01     # L2 regularization coefficient

# ---------------- Paths ----------------
MODEL_DIR = "./models"  # Directory to save fine-tuned model weights
DATA_DIR = "./data"     # Optional: local cache for downloaded datasets
